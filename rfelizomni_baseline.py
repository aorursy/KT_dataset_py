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
def baseline_filter(m_train, s_input, k=10):

    item_baselines = m_train.mean(axis=0)

    top_items = sorted(enumerate(item_baselines), key=lambda p: p[1], reverse=True)

    rec_scores = []

    for likes in s_input:

        unliked = list(filter(lambda p: p[0] not in likes, top_items))

        rec_scores.append(unliked[0:k])

    return rec_scores
k_top = 10

print("Strategy: Baseline")

rec_scores = baseline_filter(m_train, s_input)

print("MAP = {0:.3f}".format(mapk_score(s_hidden, get_recs(rec_scores), k=k_top)))

print("UHR = {0:.3f}".format(uhr_score(s_hidden, get_recs(rec_scores), k=k_top)))
def apk_scores(s_hidden, recs_pred, k=10):

    apks = []

    for r_true, r_pred in zip(s_hidden, recs_pred):

        apk = mapk_score([r_true], [r_pred], k=k)

        apks.append(apk)

    return apks
def hit_counts(s_hidden, recs_pred, k=10):

    hits = []

    for r_true, r_pred in zip(s_hidden, recs_pred):

        ix = r_true.intersection(set(r_pred[0:k]))

        hits.append(len(ix))

    return hits
apks = apk_scores(s_hidden, get_recs(rec_scores), k=k_top)

sns.distplot(apks, kde=False)

plt.xlabel("Average Precision of Recommendations for a Single User")

plt.ylabel("Number of Users")

plt.show()
hit_cts = hit_counts(s_hidden, get_recs(rec_scores), k=k_top)

sns.distplot(hit_cts, kde=False)

plt.xlabel("Number of Hits for a Single User")

plt.ylabel("Number of Users")

plt.show()
def show_user_recs(all_recs, uid):

    s_pred = get_recs(all_recs)

    print("Model: Collaborative Filtering with Jaccard Similarity (j=30)")

    print("User: {}".format(uid))

    print()

    print("Given:       {}".format(sorted(s_input[uid])))

    print("Recommended: {}".format(sorted(s_pred[uid])))

    print("Actual:      {}".format(sorted(s_hidden[uid])))

    set_intersect = set(s_pred[uid]).intersection(set(s_hidden[uid]))

    n_intersect = len(set_intersect)

    n_union = len(set(s_pred[uid]).union(set(s_hidden[uid])))

    apk = mapk_score([s_hidden[uid]], [s_pred[uid]], k_top)

    jacc = jaccard_sim(set(s_pred[uid]), set(s_hidden[uid]))

    print()

    print("Recommendation Hits = {}".format(n_intersect))

    print("Average Precision   = {0:.3f}".format(apk))

    print("Jaccard Similarity  = {0:.3f}".format(jacc))

    print()

    print("Successful Recommendations:")

    for item_id in set_intersect:

        print("- {} ({})".format(ITEM_NAMES[item_id], "cameo.com/" + ITEM_IDS[item_id]))

    print()

    print("All Recommendation Scores:")

    for i, (item_id, score) in enumerate(all_recs[uid]):

        hit = "Y" if item_id in s_hidden[uid] else " "

        print("{0}. [{3}] ({2:.3f}) {1}".format(str(i + 1).zfill(2), ITEM_NAMES[item_id], score, hit))
print("Highest AP@K: User {}".format(np.argmax(apks)))

show_user_recs(rec_scores, np.argmax(apks))
print("Most Hits: User {}".format(np.argmax(hit_cts)))

show_user_recs(rec_scores, np.argmax(hit_cts))
from redcarpet import write_kaggle_recs
# Load hold out set

s_hold_input = pickle.load(open("../input/hold_set.pkl", "rb"))

print("Hold Out Set: N = {}".format(len(s_hold_input)))

s_all_input = s_input + s_hold_input

print("All Input:    N = {}".format(len(s_all_input)))
print("Final Model")

print("Strategy: Baseline")

# Be sure to use the entire s_input

final_scores = baseline_filter(m_train, s_all_input)

final_recs = get_recs(final_scores)
outfile = "kaggle_submission_baseline.csv"

n_lines = write_kaggle_recs(final_recs, outfile)

print("Wrote predictions for {} users to {}.".format(n_lines, outfile))