import pickle

import pandas as pd

import numpy as np
with open("/kaggle/input/covid-tfidf/i_index.pickle", "rb") as f:

    i_index = pickle.load(f)
with open("/kaggle/input/covid-tfidf/tfidf_matrix.pickle", "rb") as f:

    tfidf_matrix = pickle.load(f)
with open("/kaggle/input/covid-tfidf/doc_ids.pickle", "rb") as f:

    doc_ids = pickle.load(f)
with open("/kaggle/input/covid-tfidf/feature_names.pickle", "rb") as f:

    feature_names = pickle.load(f)
tfidf_matrix.shape
tfidf_matrix[0]
def weights_for_doc(doc_idx):

    feature_index = tfidf_matrix[doc_idx , :].nonzero()[1]

    tfidf_scores = zip(feature_index, [tfidf_matrix[doc_idx, x] for x in feature_index])

    return tfidf_scores
tfidf_scores = weights_for_doc(1)

count = 0

for w, s in [(feature_names[i], s) for (i, s) in tfidf_scores]:

    n_s = np.float32(s)

    print(f"{w}: {n_s}")

    count += 1

    if (count > 10):

        break

patient_index = i_index["patient"]

patient_index
patient_index[0]
p_doc_id = int(patient_index[0][0])

p_doc_id
p_cord_id = doc_ids[p_doc_id]

p_cord_id
df_metadata = pd.read_csv("/kaggle/input/CORD-19-research-challenge/metadata.csv")

df_metadata.head()
df_metadata[df_metadata["cord_uid"] == p_cord_id]
patient_index[0][1]