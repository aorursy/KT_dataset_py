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
def get_all_scores(rec_scores, k=10):

    all_scores = []

    for recs in rec_scores:

        for (item, score) in recs[0:k]:

            all_scores.append(score)

    return all_scores
n_pred = 100 # len(s_input)

k_top = 10

j_neighbors = 30

s_input_sample = s_input[0:n_pred]

s_hidden_sample = s_hidden[0:n_pred]
print("Strategy: Collaborative")

print("Similarity: Jaccard")

collab_jac = collaborative_filter(s_train, s_input_sample, sim_fn=jaccard_sim, j=j_neighbors)

print("MAP = {0:.3f}".format(mapk_score(s_hidden_sample, get_recs(collab_jac), k=k_top)))

print("UHR = {0:.3f}".format(uhr_score(s_hidden_sample, get_recs(collab_jac), k=k_top)))
from scipy.sparse import csc_matrix

from scipy.sparse.linalg import svds
help(svds)
def svd_filter(m_train, m_input, n_factors=2, baselines=None, threshold=0, k=10):

    """

    Matrix factorization recommender system using Singular Value Decomposition (SVD).

    params:

        m_train: matrix of train data, rows = users, columns = items, 1 = like, 0 otherwise

        m_input: matrix of input data, rows = users, columns = items, 1 = like, 0 otherwise

        n_factors: number of latent factors to estimate (default: 2)

        baselines: numpy array of values to fill empty input entries, array length

            equal to the number of columns in `m_train` and `m_input` (default: None)

        threshold: minimum score to qualify as a recommended item

        k: number of items to recommend for each user

        sim_fn(u, v): function that returns a float value representing

            the similarity between sets u and v

    returns:

        recs_pred: list of lists of tuples of recommendations where

            each tuple has (item index, relevance score) with the list

            of tuples sorted in order of decreasing relevance

        (u, s, vt): tuple of matrix factors produced by SVD:

            u: latent user matrix, rows = users, columns = latent factors

            s: array of latent factor weights

            vt: transposed latent item matrix, rows = latent factors, columns = items

            To estimate the matrix, compute: `B = u.dot(np.diag(s)).dot(vt)`

    """

    m_input_filled = m_input

    if baselines is not None:

        m_input_filled = m_input + np.vstack([baselines for i in range(len(m_input))])

        m_input_filled = np.clip(m_input_filled, a_min=0, a_max=1)

    m_all = np.vstack([m_train, m_input_filled])

    A = csc_matrix(m_all, dtype=float)

    u, s, vt = svds(A, k=n_factors)

    B = u.dot(np.diag(s)).dot(vt)

    s_b_test = []

    row_base = len(m_train)

    for row_add in range(len(m_input)):

        inp = m_input[row_add]

        row_id = row_base + row_add

        row = B[row_id]

        rec_scores = []

        for col, (score, orig) in enumerate(zip(row, inp)):

            if orig < 1:

                if score >= threshold:

                    rec_scores.append((col, score))

        ranks = sorted(rec_scores, key=lambda p: p[1], reverse=True)

        s_b_test.append(ranks[0:k])

    return s_b_test, (u, s, vt)
s_b_test, (u, s, vt) = svd_filter(m_train, m_input, n_factors=2)

print("MAP = {0:.3f}".format(mapk_score(s_hidden, get_recs(s_b_test), k=10)))

print("UHR = {0:.3f}".format(uhr_score(s_hidden, get_recs(s_b_test), k=10)))

print("U: {}".format(u.shape))

print("S: {}".format(s.shape))

print("V: {}".format(vt.shape))
s_b_test, _ = svd_filter(m_train, m_input, n_factors=2, k=20)

print("MAP = {0:.3f}".format(mapk_score(s_hidden, get_recs(s_b_test), k=10)))

print("UHR = {0:.3f}".format(uhr_score(s_hidden, get_recs(s_b_test), k=10)))
sns.distplot(get_all_scores(s_b_test, k=10), kde=False, label="Top 10 Items")

sns.distplot(get_all_scores(s_b_test, k=20), kde=False, label="Top 20 Items")

plt.xlabel("Bin of Recommended Item Score")

plt.ylabel("Number of Items in Score Bin")

plt.title("Score Distribution for SVD(nf=2)")

plt.show()
s_b_test, _ = svd_filter(m_train, m_input, n_factors=2, threshold=0.01)

print("MAP = {0:.3f}".format(mapk_score(s_hidden, get_recs(s_b_test), k=10)))

print("UHR = {0:.3f}".format(uhr_score(s_hidden, get_recs(s_b_test), k=10)))
s_b_test, _ = svd_filter(m_train, m_input, n_factors=2, baselines=m_train.mean(axis=0))

print("MAP = {0:.3f}".format(mapk_score(s_hidden, get_recs(s_b_test), k=10)))

print("UHR = {0:.3f}".format(uhr_score(s_hidden, get_recs(s_b_test), k=10)))
halfs = 0.5 * np.ones(m_train.shape[1])

s_b_test, _ = svd_filter(m_train, m_input, n_factors=2, baselines=halfs)

print("MAP = {0:.3f}".format(mapk_score(s_hidden, get_recs(s_b_test), k=10)))

print("UHR = {0:.3f}".format(uhr_score(s_hidden, get_recs(s_b_test), k=10)))
nf_vals = [2, 4, 6]

maps = []

uhrs = []

for nf in nf_vals:

    s_b_test, _ = svd_filter(m_train, m_input, n_factors=nf)

    m0 = mapk_score(s_hidden, get_recs(s_b_test), k=10)

    u0 = uhr_score(s_hidden, get_recs(s_b_test), k=10)

    maps.append(m0)

    uhrs.append(u0)
sns.lineplot(nf_vals, maps)

plt.xticks(nf_vals)

plt.ylim((0, 1.1 * np.array(maps).max()))

plt.xlabel("Number of Latent Factors")

plt.ylabel("Test Set MAP Score")

plt.show()
sns.lineplot(nf_vals, uhrs)

plt.xticks(nf_vals)

plt.ylim((0, 1.1 * np.array(uhrs).max()))

plt.xlabel("Number of Latent Factors")

plt.ylabel("Test Set UHR Score")

plt.show()
from redcarpet import write_kaggle_recs
# Load hold out set

s_hold_input = pickle.load(open("../input/hold_set.pkl", "rb"))

print("Hold Out Set: N = {}".format(len(s_hold_input)))

s_all_input = s_input + s_hold_input

print("All Input:    N = {}".format(len(s_all_input)))
# Load hold out set

m_hold_input = pickle.load(open("../input/hold_mat.pkl", "rb"))

m_hold_input = np.array(m_hold_input.todense())

print("Hold Out Set: N = {}".format(len(m_hold_input)))

m_all_input = np.vstack([m_input, m_hold_input])

print("All Input:    N = {}".format(len(m_all_input)))
print("Final Model")

print("Strategy: SVD(nf=8)")

# Be sure to use the entire s_input

final_scores, _ = svd_filter(m_train, m_all_input, n_factors=8)

final_recs = get_recs(final_scores, k=10)

len(final_recs)
outfile = "kaggle_submission_svd_nf8_fill.csv"

n_lines = write_kaggle_recs(final_recs, outfile)

print("Wrote predictions for {} users to {}.".format(n_lines, outfile))
from IPython.display import HTML

import base64





def download_kaggle_recs(recs_list, filename=None, headers=["Id", "Predicted"]):

    """

    Writes recommendations to file in Kaggle submission format.

    params:

        recs_list: list of lists of recommendations where each

            list has the column indices of recommended items

            sorted in order of decreasing relevance

        filename: path to file for writing output

        headers: list of strings of output columns, defaults to

            submission columns: ["Id", "Predicted"]

    returns:

        html: HTML download link to display in a notebook, click

            to download the submission file

    """

    if filename is None:

        raise ValueError("Must provide a filename.")

    rec_df = pd.DataFrame(

        [(i, " ".join([str(r) for r in recs])) for i, recs in enumerate(recs_list)],

        columns=headers,

    )

    csv = rec_df.to_csv(index=False)

    b64 = base64.b64encode(csv.encode())

    payload = b64.decode()

    html = """<a download="{filename}"

                href="data:text/csv;base64,{payload}"

                target="_blank">Download ({lines} lines): {filename}</a>"""

    html = html.format(payload=payload, filename=filename, lines=len(rec_df))

    return HTML(html)
download_kaggle_recs(final_recs, outfile)