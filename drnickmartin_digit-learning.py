import numpy as np

import pandas as pd

from sklearn import svm

from sklearn import preprocessing



# Read in the data

train = pd.read_csv(r'../input/train.csv')

test = pd.read_csv(r'../input/test.csv')

Xtrain = train.as_matrix()[:,1:]

Xtest = test.as_matrix()

y = train.as_matrix()[:,0]



# Normalise the datasets

scaler = preprocessing.StandardScaler().fit(Xtrain)

XtrainN = scaler.transform(Xtrain)

XtestN = scaler.transform(Xtest)



# Train the svm and fit the data

clf = svm.SVC(kernel='rbf')

clf.fit(XtrainN, y)

res = clf.predict(XtestN)



# Output a dataframe for submission

pred = pd.DataFrame()

pred["ImageId"] = np.arange(1,res.shape[0]+1)

pred["Label"] = res

pred.to_csv(r'predictions.csv')