import pandas as pd

import numpy as np

%matplotlib inline

import matplotlib.pyplot as plt

import seaborn as sns

import pickle

import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)
# Load module from another directory

import shutil

shutil.copyfile(src="../input/redcarpet.py", dst="../working/redcarpet.py")

from redcarpet import mat_to_sets
item_file = "../input/talent.pkl"

item_records, COLUMN_LABELS, READABLE_LABELS, ATTRIBUTES = pickle.load(open(item_file, "rb"))

item_df = pd.DataFrame(item_records)[ATTRIBUTES + COLUMN_LABELS].fillna(value=0)

ITEM_NAMES = item_df["name"].values

ITEM_IDS = item_df["id"].values

item_df.head()
s_items = mat_to_sets(item_df[COLUMN_LABELS].values)

print("Items", len(s_items))

csr_train, csr_test, csr_input, csr_hidden = pickle.load(open("../input/train_test_mat.pkl", "rb"))

m_split = [np.array(csr.todense()) for csr in [csr_train, csr_test, csr_input, csr_hidden]]

m_train, m_test, m_input, m_hidden = m_split

print("Matrices", len(m_train), len(m_test), len(m_input), len(m_hidden))

s_train, s_test, s_input, s_hidden = pickle.load(open("../input/train_test_set.pkl", "rb"))

print("Sets", len(s_train), len(s_test), len(s_input), len(s_hidden))
like_df = pd.DataFrame(m_train, columns=ITEM_NAMES)

like_df.head()
from redcarpet import mapk_score, uhr_score

from redcarpet import jaccard_sim, cosine_sim

from redcarpet import collaborative_filter, content_filter, weighted_hybrid

from redcarpet import get_recs
from mlxtend.frequent_patterns import apriori
help(apriori)
def mine_association_rules(m_train, min_support=0.5):

    freq_is = apriori(pd.DataFrame(m_train), max_len=2, min_support=min_support)

    freq_is["len"] = freq_is["itemsets"].apply(lambda s: len(s))

    freq_is = freq_is.query("len == 2")

    if len(freq_is) == 0:

        return pd.DataFrame([], columns=["a", "b", "ct", "support"])

    item_counts = m_train.sum(axis=0)

    rules = []

    for record in freq_is.to_dict(orient="records"):

        fset = record["itemsets"]

        a = min(fset)

        b = max(fset)

        n = len(m_train)

        supp = record["support"]

        all_a = item_counts[a]

        all_b = item_counts[b]

        both = supp * n

        f11 = int(both)

        f10 = int(all_a - both)

        f01 = int(all_b - both)

        f00 = int(n - (f11 + f10 + f01))

        rules.append({"a": a, "b": b, "ct": (f11, f10, f01, f00), "support": supp})

        rules.append({"a": b, "b": a, "ct": (f11, f01, f10, f00), "support": supp})

    rule_df = pd.DataFrame(rules)

    return rule_df
all_rules = mine_association_rules(m_train, min_support=0.025)

print("Found {} association rules.".format(len(all_rules)))

all_rules.sort_values(by=["support"], ascending=False).head()
supps = all_rules["support"].values

sns.distplot(supps[supps > 0], kde=False, bins=np.arange(0, max(supps), 0.01))

plt.xlabel("Support")

plt.ylabel("Number of Rules")

plt.show()
used_rules = all_rules.query("support >= 0.03")

len(all_rules), len(used_rules)
def sets_to_contingency(a, b, N):

    f11 = len(a.intersection(b))

    f10 = len(a) - f11

    f01 = len(b) - f11

    f00 = N - (f11 + f10 + f01)

    return f11, f10, f01, f00



# https://github.com/resumesai/resumesai.github.io/blob/master/analysis/Rule%20Mining.ipynb



def rule_support(f11, f10, f01, f00):

    N = f11 + f10 + f01 + f00

    return f11 / N



def rule_confidence(f11, f10, f01, f00):

    return f11 / (f11 + f10)



def rule_interest_factor(f11, f10, f01, f00):

    N = f11 + f10 + f01 + f00

    f1p = f11 + f10

    fp1 = f11 + f01

    return (N * f11) / (f1p * fp1)



def rule_phi_correlation(f11, f10, f01, f00):

    f1p = f11 + f10

    f0p = f01 + f00

    fp1 = f11 + f01

    fp0 = f10 + f00

    num = (f11 * f00) - (f01 * f10)

    denom = np.sqrt(f1p * fp1 * f0p * fp0)

    if denom == 0:

        return 0.0

    return num / denom



def rule_is_score(f11, f10, f01, f00):

    intfac = rule_interest_factor(f11, f10, f01, f00)

    supp = support(f11, f10, f01, f00)

    return np.sqrt(intfac * supp)
def rank_association_rules(mined_rules_df, score_fn, score_name="score"):

    rule_df = pd.DataFrame(mined_rules_df.copy())

    rule_df[score_name] = rule_df["ct"].apply(lambda ct: score_fn(*ct))

    return rule_df.sort_values(by=score_name, ascending=False)
def association_filter(rules_df, m_train, s_input, score_fn=rule_support, min_score=0.01, k=10):

    score_name = "score"

    ranked_rules = rank_association_rules(rules_df, score_fn=score_fn, score_name=score_name)

    top_rules_df = ranked_rules.query("{} >= {}".format(score_name, min_score))

    rule_records = top_rules_df.to_dict(orient="records")

    all_recs = []

    for likes in s_input:

        rec_map = {}

        for rule in rule_records:

            if rule["a"] in likes and rule["b"] not in likes:

                if rule["b"] not in rec_map:

                    rec_map[rule["b"]] = 0

                rec_map[rule["b"]] += rule[score_name]

        ranks = sorted(rec_map.items(), key=lambda p: p[1], reverse=True)

        all_recs.append(ranks[0:k])

    return all_recs, top_rules_df
k_top = 10

print("Metric: Support")

rec_scores, rule_df = association_filter(used_rules, m_train, s_input, score_fn=rule_support)

print("MAP = {0:.3f}".format(mapk_score(s_hidden, get_recs(rec_scores), k=k_top)))

print("UHR = {0:.3f}".format(uhr_score(s_hidden, get_recs(rec_scores), k=k_top)))

print("Used {} association rules.".format(len(rule_df)))

rule_df.head()
print("Metric: Confidence")

rec_scores, rule_df = association_filter(used_rules, m_train, s_input, score_fn=rule_confidence)

print("MAP = {0:.3f}".format(mapk_score(s_hidden, get_recs(rec_scores), k=k_top)))

print("UHR = {0:.3f}".format(uhr_score(s_hidden, get_recs(rec_scores), k=k_top)))

print("Used {} association rules.".format(len(rule_df)))

rule_df.head()
print("Metric: Phi Correlation")

rec_scores, rule_df = association_filter(used_rules, m_train, s_input, score_fn=rule_phi_correlation)

print("MAP = {0:.3f}".format(mapk_score(s_hidden, get_recs(rec_scores), k=k_top)))

print("UHR = {0:.3f}".format(uhr_score(s_hidden, get_recs(rec_scores), k=k_top)))

print("Used {} association rules.".format(len(rule_df)))

rule_df.head()
print("Metric: IS Score")

rec_scores, rule_df = association_filter(used_rules, m_train, s_input, score_fn=rule_is_score)

print("MAP = {0:.3f}".format(mapk_score(s_hidden, get_recs(rec_scores), k=k_top)))

print("UHR = {0:.3f}".format(uhr_score(s_hidden, get_recs(rec_scores), k=k_top)))

print("Used {} association rules.".format(len(rule_df)))

rule_df.head()
def get_all_scores(rec_scores):

    all_scores = []

    for recs in rec_scores:

        for (item, score) in recs:

            all_scores.append(score)

    return all_scores
supp_scores, _ = association_filter(used_rules, m_train, s_input, score_fn=rule_support)

conf_scores, _ = association_filter(used_rules, m_train, s_input, score_fn=rule_confidence)

phi_scores, _ = association_filter(used_rules, m_train, s_input, score_fn=rule_phi_correlation)

is_scores, _ = association_filter(used_rules, m_train, s_input, score_fn=rule_is_score)
sns.distplot(get_all_scores(supp_scores), kde=False, label="Support")

sns.distplot(get_all_scores(conf_scores), kde=False, label="Confidence")

sns.distplot(get_all_scores(phi_scores), kde=False, label="Phi Correlation")

sns.distplot(get_all_scores(is_scores), kde=False, label="IS Score")

plt.xlabel("Score")

plt.ylabel("Count")

plt.title("Association Rule Score Distributions")

plt.legend()

plt.show()
rec_scores = weighted_hybrid([

    (supp_scores, 0.25),

    (conf_scores, 0.25),

    (phi_scores, 0.25),

    (is_scores, 0.25),

])

print("MAP = {0:.3f}".format(mapk_score(s_hidden, get_recs(rec_scores), k=k_top)))

print("UHR = {0:.3f}".format(uhr_score(s_hidden, get_recs(rec_scores), k=k_top)))
sns.distplot(get_all_scores(rec_scores), kde=False)

plt.xlabel("Score")

plt.ylabel("Count")

plt.title("Weighted Hybrid Score Distribution")

plt.show()
rec_scores = weighted_hybrid([

    (supp_scores, 2.0),

    (conf_scores, 0.25),

    (phi_scores, 0.25),

    (is_scores, 0.25),

])

print("MAP = {0:.3f}".format(mapk_score(s_hidden, get_recs(rec_scores), k=k_top)))

print("UHR = {0:.3f}".format(uhr_score(s_hidden, get_recs(rec_scores), k=k_top)))
rec_scores = weighted_hybrid([

    (supp_scores, 3.0),

    (conf_scores, 1.5),

    (phi_scores, 1.5),

    (is_scores, 1.0),

])

print("MAP = {0:.3f}".format(mapk_score(s_hidden, get_recs(rec_scores), k=k_top)))

print("UHR = {0:.3f}".format(uhr_score(s_hidden, get_recs(rec_scores), k=k_top)))
cos_scores = collaborative_filter(s_train, s_input, sim_fn=cosine_sim, j=30)

print("MAP = {0:.3f}".format(mapk_score(s_hidden, get_recs(cos_scores), k=k_top)))

print("UHR = {0:.3f}".format(uhr_score(s_hidden, get_recs(cos_scores), k=k_top)))
len(cos_scores), len(is_scores)
rec_scores = weighted_hybrid([

    (supp_scores, 1.0),

    (conf_scores, 1.0),

    (phi_scores, 1.0),

    (is_scores, 1.0),

    (cos_scores, 10.0)

])

print("MAP = {0:.3f}".format(mapk_score(s_hidden, get_recs(rec_scores), k=k_top)))

print("UHR = {0:.3f}".format(uhr_score(s_hidden, get_recs(rec_scores), k=k_top)))
from redcarpet import write_kaggle_recs
# Load hold out set

s_hold_input = pickle.load(open("../input/hold_set.pkl", "rb"))

print("Hold Out Set: N = {}".format(len(s_hold_input)))

s_all_input = s_input + s_hold_input

print("All Input:    N = {}".format(len(s_all_input)))
print("Final Model")

print("Strategy: Association Rules")

print("Scoring: Hybrid")

# Be sure to use the entire s_input

# cos_scores = collaborative_filter(s_train, s_all_input, sim_fn=cosine_sim, j=30)

supp_scores, _ = association_filter(used_rules, m_train, s_all_input, score_fn=rule_support)

conf_scores, _ = association_filter(used_rules, m_train, s_all_input, score_fn=rule_confidence)

phi_scores, _ = association_filter(used_rules, m_train, s_all_input, score_fn=rule_phi_correlation)

is_scores, _ = association_filter(used_rules, m_train, s_all_input, score_fn=rule_is_score)

final_scores = weighted_hybrid([

    (supp_scores, 0.25),

    (conf_scores, 0.25),

    (phi_scores, 0.25),

    (is_scores, 0.25),

])

final_recs = get_recs(final_scores)
outfile = "kaggle_submission_association_rules_hybrid.csv"

n_lines = write_kaggle_recs(final_recs, outfile)

print("Wrote predictions for {} users to {}.".format(n_lines, outfile))