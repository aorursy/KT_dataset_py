#from google.colab import files

#uploader = files.upload()
import pandas as pd



trainingData = pd.read_csv("/kaggle/input/data-mining-assignment-2/train.csv")

trainingData.head()
df = trainingData.copy()

df.drop(['ID'], axis = 1, inplace=True)

hotEncodedDF = pd.get_dummies(data=df, columns=["col2","col11","col37","col44","col56"])
from sklearn import preprocessing as pp

scalar = pp.MinMaxScaler()

YAxis = hotEncodedDF['Class']

XAxis = hotEncodedDF.drop('Class', axis=1)

scaledTransform = scalar.fit_transform(XAxis)

XAxisDF = pd.DataFrame(scaledTransform)

XAxisDF.head()
from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

from sklearn.ensemble import RandomForestClassifier





xTrain, xTest, yTrain, yTest = train_test_split(XAxisDF, YAxis, test_size=0.25, random_state=169)

trainScore = []

testScore = []



for i in range(0,15):

    rf = RandomForestClassifier(max_depth=i + 5, n_estimators=150)

    rf.fit(xTrain, yTrain)

    trainScoreData = rf.score(xTrain, yTrain)

    trainScore.append(trainScoreData)

    testScoreData = rf.score(xTest, yTest)

    testScore.append(testScoreData)
rf = RandomForestClassifier(n_estimators=125, max_depth = 8)

rf.fit(xTrain, yTrain)

rf.score(xTest,yTest)




rf = RandomForestClassifier(n_estimators=2500, max_depth = 10)

rf.fit(xTrain, yTrain)

print(rf.score(xTest, yTest))

yPrediction = rf.predict(xTest)

confusion_matrix(yTest, yPrediction)
rFset = RandomForestClassifier(n_estimators=100, min_samples_split=5, max_depth=6)

rFset.fit(xTrain, yTrain)

rFset.score(xTest, yTest)

yPredictionBest = rFset.predict(xTest)

confusion_matrix(yTest, yPredictionBest)
testData = pd.read_csv("/kaggle/input/data-mining-assignment-2/test.csv")

testData = pd.get_dummies(testData, columns=["col2","col11","col37","col44","col56"])

testData.drop('ID',axis=1,inplace=True)

testData.head()
from sklearn import preprocessing

scalar = preprocessing.MinMaxScaler()

scaledData = scalar.fit_transform(testData)

testDF = pd.DataFrame(scaledData)

testDF.head()
yPredictionBest = rFset.predict(testDF)

yPredictionBest
answers = pd.DataFrame(0, index=range(300), columns=['ID','Class'])

for i in range(0, 300):

    answers['ID'][i] = i + 700

    answers['Class'][i] = yPredictionBest[i]



answers.to_csv('answer.csv',index=False)

answers
from IPython.display import HTML

import pandas as pd

import numpy as np

import base64

def create_download_link(df, title = "Download CSV file", filename = "answer.csv"):

 csv = df.to_csv(index=False)

 b64 = base64.b64encode(csv.encode())

 payload = b64.decode()

 html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'

 html = html.format(payload=payload,title=title,filename=filename)

 return HTML(html)

create_download_link(answers)