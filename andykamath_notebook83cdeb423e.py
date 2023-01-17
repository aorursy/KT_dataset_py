#Import all required files

from sklearn.neural_network import MLPClassifier

import h5py

from scipy import sparse

import numpy as np

from sklearn.model_selection import cross_val_score

print("Modules imported!")
#Collect data including features and labels

print("Collecting Data...")

hf = h5py.File("../input/cdk2.h5", "r")

ids = hf["chembl_id"].value # the name of each molecules

ap = sparse.csr_matrix((hf["ap"]["data"], hf["ap"]["indices"], hf["ap"]["indptr"]), shape=[len(hf["ap"]["indptr"]) - 1, 2039])

mg = sparse.csr_matrix((hf["mg"]["data"], hf["mg"]["indices"], hf["mg"]["indptr"]), shape=[len(hf["mg"]["indptr"]) - 1, 2039])

tt = sparse.csr_matrix((hf["tt"]["data"], hf["tt"]["indices"], hf["tt"]["indptr"]), shape=[len(hf["tt"]["indptr"]) - 1, 2039])

features = sparse.hstack([ap, mg, tt]).toarray() # the samples' features, each row is a sample, and each sample has 3*2039 features

labels = hf["label"].value # the label of each molecule

print("Data collected. Training ANN...")
#Define ANN Architecture

X_train, X_test, y_train, y_test = [features[:-100], features[-100:], labels[:-100], labels[-100:]]

ann = MLPClassifier(verbose=True, warm_start=True, max_iter=200)

ann.fit(X_train, y_train)

scores = cross_val_score(ann, features, labels)