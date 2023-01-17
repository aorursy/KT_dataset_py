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

s_items = mat_to_sets(item_df[COLUMN_LABELS].values)

assert len(item_df) == len(s_items), "Item matrix is not the same length as item category set list."

print("Talent:", len(item_df))

print("Categories:", len(COLUMN_LABELS))

item_df.head()
def cameo_name(i):

    """

    Show the name and URL of Cameo talent based on its index `i`.

    """

    return "{} (cameo.com/{})".format(ITEM_NAMES[i], ITEM_IDS[i])
csr_train, csr_test, csr_input, csr_hidden = pickle.load(open("../input/train_test_mat.pkl", "rb"))

m_split = [np.array(csr.todense()) for csr in [csr_train, csr_test, csr_input, csr_hidden]]

m_train, m_test, m_input, m_hidden = m_split

s_train, s_test, s_input, s_hidden = pickle.load(open("../input/train_test_set.pkl", "rb"))

assert len(m_train) == len(s_train), "Train matrix is not the same length as train sets."

assert len(m_test) == len(s_test), "Test matrix is not the same length as test sets."

assert len(m_input) == len(s_input), "Input matrix is not the same length as input sets."

assert len(m_hidden) == len(s_hidden), "Hidden matrix is not the same length as hidden sets."

print("Train Users", len(m_train))

print("Test Users", len(m_test))

print("Minimum Test Items per User:", min(m_test.sum(axis=1)))

print("Minimum Input Items per User:", min(m_input.sum(axis=1)))

print("Minimum Hidden Items per User:", min(m_hidden.sum(axis=1)))

like_df = pd.DataFrame(m_train, columns=ITEM_NAMES)

like_df.head()
from redcarpet import mapk_score, uhr_score
help(mapk_score)
help(uhr_score)
from redcarpet import jaccard_sim, cosine_sim
help(jaccard_sim)
help(cosine_sim)
from redcarpet import collaborative_filter, content_filter, weighted_hybrid
help(collaborative_filter)
help(content_filter)
help(weighted_hybrid)
from redcarpet import get_recs

from redcarpet import show_user_recs, show_item_recs, show_user_detail

from redcarpet import show_apk_dist, show_hit_dist, show_score_dist
help(get_recs)
k_top = 10
print("Model: Collaborative Filtering with Jacccard Similarity (j=10)")

collab_jac10 = collaborative_filter(s_train, s_input, sim_fn=jaccard_sim, threshold=0.01,j=50, k=k_top)

print("MAP = {0:.3f}".format(mapk_score(s_hidden, get_recs(collab_jac10), k=k_top)))

print("UHR = {0:.3f}".format(uhr_score(s_hidden, get_recs(collab_jac10), k=k_top)))
idf = show_item_recs(s_hidden, collab_jac10, k=k_top)

idf.sort_values(by=["Hits", "Hit Rate"], ascending=[False, False]).head()
udf = show_user_recs(s_hidden, collab_jac10, k=k_top)

udf.sort_values(by=["APK", "Hits"], ascending=[False, False]).head()
show_user_detail(s_input, s_hidden, collab_jac10, uid=0, name_fn=cameo_name)
help(collaborative_filter)
print("Model: Collaborative Filtering with Cosine Similarity (j=10)")

collab_cos10 = collaborative_filter(s_train, s_input, sim_fn=cosine_sim, threshold = 0.1,j=70, k=k_top)

print("MAP = {0:.3f}".format(mapk_score(s_hidden, get_recs(collab_cos10), k=k_top)))

print("UHR = {0:.3f}".format(uhr_score(s_hidden, get_recs(collab_cos10), k=k_top)))
results = [

    (collab_jac10, "Jaccard (j=10)"),

    (collab_cos10, "Cosine (j=10)")

]
show_apk_dist(s_hidden, results, k=k_top)
show_hit_dist(s_hidden, results, k=k_top)
show_score_dist(results, k=10, bins=np.arange(0.0, 1.1, 0.1))
print("Model: Hybrid Collaborative Filtering")

print("Similarity: Hybrid (0.2 * Jaccard + 0.8 * Cosine)")

collab_hybrid = weighted_hybrid([

    (collab_jac10, 0.3),

    (collab_cos10, 0.7)

])

print("MAP = {0:.3f}".format(mapk_score(s_hidden, get_recs(collab_hybrid), k=k_top)))

print("UHR = {0:.3f}".format(uhr_score(s_hidden, get_recs(collab_hybrid), k=k_top)))
from redcarpet import write_kaggle_recs, download_kaggle_recs
# Load hold out set

s_hold_input = pickle.load(open("../input/hold_set.pkl", "rb"))

print("Hold Out Set: N = {}".format(len(s_hold_input)))

s_all_input = s_input + s_hold_input

print("All Input:    N = {}".format(len(s_all_input)))
print("Final Model")

print("Strategy: Collaborative")

print("Similarity: Cosine (j=10)")

# Be sure to use the entire s_input

final_scores = collaborative_filter(s_train, s_all_input, sim_fn=cosine_sim, j=10)

final_recs = get_recs(final_scores)
outfile = "kaggle_submission_hybrid_collab.csv"

n_lines = write_kaggle_recs(final_recs, outfile)

print("Wrote predictions for {} users to {}.".format(n_lines, outfile))

download_kaggle_recs(final_recs, outfile)