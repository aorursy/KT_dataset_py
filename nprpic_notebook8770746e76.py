# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import sklearn as skl

import seaborn as sns

import matplotlib.pyplot as plt



data = pd.read_csv("../input/Iris.csv")

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



# from subprocess import check_output

# print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
data.head()



data["Species"].value_counts()
data.iloc[:,1:5].plot(kind="kde", subplots= True)



# sns.FacetGrid(data, hue="Species").map(sns.kdeplot, "SepalLengthCm").add_legend()
data.sort_values(['Species'])

setosa=data.iloc[0:50,:]

virginica=data.iloc[51:100,:]

versicolor=data.iloc[101:150,:]



sns.pairplot(data.drop("Id", axis=1), hue="Species", size=5, diag_kind="kde")
from sklearn import svm

from sklearn.model_selection import train_test_split as tts



train, test= tts(data, test_size=0.2)



trainX=train.iloc[:,1:5].values

trainY=train.iloc[:,5].values

testX=test.iloc[:,1:5].values

testY=test.iloc[:,5].values



svc = svm.SVC()

svc.fit(trainX,trainY)

svc.score(testX,testY)
from sklearn.neural_network import MLPClassifier as MLPC



SCORESt=[]

for j in range (1,20):

    SCORE=0

    for i in range (0,30):

        train, test = tts(data, test_size=0.2)

        trainX=train.iloc[:,1:5].values

        trainY=train.iloc[:,5].values

        testX=test.iloc[:,1:5].values

        testY=test.iloc[:,5].values

        neur = MLPC(hidden_layer_sizes=(j), activation="tanh", solver="lbfgs")

        neur.fit(trainX, trainY)

        SCORE = SCORE + neur.score(testX, testY)

    print(SCORE/30)

    SCORESt.append(SCORE)