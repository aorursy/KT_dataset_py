# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
trainData = pd.read_csv("../input/train.csv").fillna(0)

testData = pd.read_csv("../input/test.csv").fillna(0)

print(trainData.shape)

print(testData.shape)
display(trainData.describe())

display(trainData.describe(include=["O"]))
trainData.corrwith(trainData.SalePrice)
def load_and_preprocess():

    def oneHotFrame(df):

        cat = pd.get_dummies(df.select_dtypes(include='object'))

        nonCat = df.select_dtypes(exclude='object')



        result = pd.concat([nonCat, cat], axis=1)

        return result

    

    trainData = pd.read_csv("../input/train.csv").fillna(0)

    testData = pd.read_csv("../input/test.csv").fillna(0)

    

    print("Raw data")

    print(trainData.shape)

    print(testData.shape)

    print()

    

    fullData = pd.concat([trainData, testData], sort=True)

    modifiedFullData = oneHotFrame(fullData).fillna(0)



    modifiedTrainData, modifiedTestData = modifiedFullData[:len(trainData)], modifiedFullData[len(trainData):]



    print("One-hot")

    print(modifiedTrainData.shape)

    print(modifiedTestData.shape)

    print()

    

    trainX = modifiedTrainData.drop('SalePrice', axis=1)

    trainy = modifiedTrainData.SalePrice



    testData = modifiedTestData.drop('SalePrice', axis=1)



    print("X and y")

    print(trainX.shape)

    print(trainy.shape)

    return trainX, trainy, testData



trainX, trainy, testData = load_and_preprocess()
from sklearn import tree

dtree = tree.DecisionTreeRegressor(max_depth=10)



from sklearn import linear_model

reg = linear_model.LinearRegression()

ridge = linear_model.Ridge(alpha=.5)

ridgeCV = linear_model.RidgeCV(alphas=[0.10, 0.1, 0.5, 1.0, 10.0], cv=3)
from sklearn.model_selection import cross_val_score

for estimator in [dtree, reg, ridge, ridgeCV]:

    print(type(estimator).__name__)

    print(cross_val_score(estimator, trainX, trainy, cv=5))

    print()
predictor = ridgeCV

predictor.fit(trainX,trainy)

predictions = pd.Series(predictor.predict(testData.values))
output = pd.concat([testData.Id, predictions], axis=1)

output.columns=["Id", "SalePrice"]

output.to_csv("dtree_predictions.csv", index=False)

print(os.listdir("."))