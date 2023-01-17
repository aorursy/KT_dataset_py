import os

import pandas as pd

import numpy as np

import tensorflow as tf

import matplotlib.pyplot as plt

from sklearn.decomposition.kernel_pca import KernelPCA

from sklearn.metrics import classification_report

from sklearn.preprocessing import PolynomialFeatures

from sklearn import preprocessing

from sklearn import ensemble
trainData = pd.read_csv("../input/train.csv")

testData = pd.read_csv("../input/test.csv")
trainData.isnull().sum()
testData.isnull().sum()
np.unique(trainData[['type']].values)
np.unique(trainData[['color']].values)
np.unique(testData[['color']].values)
trainData.head()
lbl = preprocessing.LabelEncoder()

lbl.fit(list(trainData['color'].values)) 

trainData['color'] = lbl.transform(list(trainData['color'].values))



lbl = preprocessing.LabelEncoder()

lbl.fit(list(trainData['type'].values)) 

trainData['type'] = lbl.transform(list(trainData['type'].values))
trainData.head()
yTrain = trainData['type'].values

xTrain = trainData.drop(["id", "type"], axis=1)

xTrain.head()
model = ensemble.RandomForestClassifier(n_estimators=170)

model.fit(xTrain, yTrain)
model.score(xTrain,yTrain)
lbl = preprocessing.LabelEncoder()

lbl.fit(list(testData['color'].values)) 

testData['color'] = lbl.transform(list(testData['color'].values))
testData.head()
yTest = testData['id'].values

xTest = testData.drop(["id"], axis=1)

xTest.head()
pred = model.predict(xTest)

my_submission = pd.DataFrame({'ID': yTest, 'y': pred})
predic = pd.read_csv('../input/sample_submission.csv')
my_submission_new = []

i = 0

for row in my_submission.iterrows():

    my = {}

    my['id'] = predic.id[i]

    if(row[1]['y'] ==0):

        my['type'] = 'Ghost'

    elif(row[1]['y'] ==1):

        my['type'] = 'Ghoul'

    else:

        my['type'] = 'Goblin'

    my_submission_new.append(my)

    i = i+1
df = pd.DataFrame(my_submission_new, columns=["id","type"])
df.to_csv('submission.csv', index=False)