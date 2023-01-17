# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import tensorflow as tf

from sklearn.metrics import classification_report, confusion_matrix

import re
trainData = pd.read_csv("/kaggle/input/nlp-getting-started/train.csv")

testData = pd.read_csv("/kaggle/input/nlp-getting-started/test.csv")

sampleData = pd.read_csv("/kaggle/input/nlp-getting-started/sample_submission.csv")
trainData.head()
pd.unique(trainData["keyword"])[:10]
def save(pred):

    d = {

        "id" : testData["id"],

        "target" : pred

    }

    pd.DataFrame(d).to_csv("/kaggle/working/submit.csv", index=False)
# May be this get 1st positon in leaderboard.

rand = np.random.randint(0, 2, size = len(testData)).tolist()

save(rand)

# score .50444 :(
# Trying some trick

prediction = [0 if k is np.nan else 1 for k in testData["keyword"]]

save(prediction)

# score .42782 :@ unexpected
# Apply some If else intersection

 

trainKeyword = pd.unique(trainData["keyword"].fillna("").values).tolist() # Unique keywords of train-set

testKeyword = pd.unique(testData["keyword"].fillna("").values).tolist() # Unique keywords of test-set

allKeyword = " ".join(pd.unique(trainKeyword+testKeyword)).lower() 

allKeyword = re.sub(r"[^a-zA-Z ]+", " ", allKeyword).strip().split() # Removing !alpha char

# Thats all I need

print(allKeyword[:10])
# Cleaning testData and spliting

testDataCleanText = testData["text"].apply(lambda x: re.sub(r"[^a-zA-Z]+", " ", x).lower().split())

# Finding count of keyword common in both list 

intersection = testDataCleanText.apply(lambda x: list(set(x).intersection(allKeyword)))

print(intersection[:8])

prediction = [0 if len(i) <2 else 1 for i in intersection]

save(prediction)

# score .64633 :) above random
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.linear_model import LogisticRegression
# Some preprocessing

allText = " ".join(trainData["text"].apply(lambda x: re.sub(r"[^a-zA-Z]+", " ", x).lower()))

allText += " ".join(testData["text"].apply(lambda x: re.sub(r"[^a-zA-Z]+", " ", x).lower()))

allText = allText.split()



print("This will be the size of training vector", end=" ")

print(len(trainData), len(np.unique(allText)))



# Using CountVectorizer

countVector = CountVectorizer().fit(allText)
# Training Vector

featureCol = trainData["text"].apply(lambda x: re.sub(r"[^a-zA-Z]+", " ", x).lower())

target = trainData["target"]

trainVector = countVector.transform(featureCol).toarray()
# Using Logistic regression

clf = LogisticRegression().fit(trainVector, target)

prediction = clf.predict(trainVector)

print("Training Report:")

print(classification_report(prediction, target))

print("Confusion Metrix:")

print(confusion_matrix(prediction, target))

# Results looking good
# Test prediction

testFeatureCol = testData["text"].apply(lambda x: re.sub(r"[^a-zA-Z]+", " ", x).lower())

testVector = countVector.transform(testFeatureCol)

testPrediction = clf.predict(testVector)

save(testPrediction)

# score 0.79865 not bad, expected more with above result