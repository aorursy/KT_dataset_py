# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

from sklearn.neural_network import MLPClassifier

import h5py

from scipy import sparse

import numpy as np

from sklearn.model_selection import cross_val_score

from sklearn.svm import SVC

print("Modules imported!")

print("Collecting Data...")

hf = h5py.File("../input/cdk2.h5", "r")

ids = hf["chembl_id"].value # the name of each molecules

ap = sparse.csr_matrix((hf["ap"]["data"], hf["ap"]["indices"], hf["ap"]["indptr"]), shape=[len(hf["ap"]["indptr"]) - 1, 2039])

mg = sparse.csr_matrix((hf["mg"]["data"], hf["mg"]["indices"], hf["mg"]["indptr"]), shape=[len(hf["mg"]["indptr"]) - 1, 2039])

tt = sparse.csr_matrix((hf["tt"]["data"], hf["tt"]["indices"], hf["tt"]["indptr"]), shape=[len(hf["tt"]["indptr"]) - 1, 2039])

features = sparse.hstack([ap, mg, tt]).toarray() # the samples' features, each row is a sample, and each sample has 3*2039 features

labels = hf["label"].value # the label of each molecule

print("Data collected. Training ANN...")

X_train, X_test, y_train, y_test = [features[:-100], features[-100:], labels[:-100], labels[-100:]]

ann = MLPClassifier(verbose=True, warm_start=True, max_iter=200)

ann = SVC()

ann.fit(X_train, y_train)

print("ANN trained. Testing ANN...")

tin = X_test

tout = y_test

tp = 0

tn = 0

fp = 0

fn = 0

for i, a in enumerate(tin):

	if ann.predict([a])[0] == tout[i]:

		if tout[i] == 1:

			tp += 1

		else:

			tn += 1

	else:

		if tout[i] == 1:

			fp += 1

		else:

			fn += 1

scores = cross_val_score(ann, features, labels, cv=5)

print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))



# Any results you write to the current directory are saved as output.