# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/smarket.csv")

df = df.drop([df.columns[0]],axis=1)

df.head()
df.boxplot(column=["Year"])
df["Year"].describe()
df["Year"].median()
df.boxplot(column=["Lag1"])
df["Lag1"].describe()
skew = (df["Lag1"].quantile(q=0.75) - df["Lag1"].quantile(q=0.5)) - (df["Lag1"].quantile(q=0.5) - df["Lag1"].quantile(q=0.25))

print("skew: "+ str(skew))
df.boxplot(column=["Lag2"])
df["Lag2"].describe()
skew = (df["Lag2"].quantile(q=0.75) - df["Lag2"].quantile(q=0.5)) - (df["Lag2"].quantile(q=0.5) - df["Lag2"].quantile(q=0.25))

print("skew: "+ str(skew))
df.boxplot(column=["Lag3"])
df["Lag3"].describe()
skew = (df["Lag3"].quantile(q=0.75) - df["Lag3"].quantile(q=0.5)) - (df["Lag3"].quantile(q=0.5) - df["Lag3"].quantile(q=0.25))

print("skew: "+ str(skew))
df.boxplot(column=["Lag4"])
df.describe()
skew = (df["Lag4"].quantile(q=0.75) - df["Lag4"].quantile(q=0.5)) - (df["Lag4"].quantile(q=0.5) - df["Lag4"].quantile(q=0.25))

print("skew: "+ str(skew))
df.boxplot(column=["Lag5"])
skew = (df["Lag5"].quantile(q=0.75) - df["Lag5"].quantile(q=0.5)) - (df["Lag5"].quantile(q=0.5) - df["Lag5"].quantile(q=0.25))

print("skew: "+ str(skew))
df.boxplot(column=["Volume"])
skew = (df["Volume"].quantile(q=0.75) - df["Volume"].quantile(q=0.5)) - (df["Volume"].quantile(q=0.5) - df["Volume"].quantile(q=0.25))

print("skew: "+ str(skew))
df.boxplot(column=["Today"])
skew = (df["Volume"].quantile(q=0.75) - df["Volume"].quantile(q=0.5)) - (df["Volume"].quantile(q=0.5) - df["Volume"].quantile(q=0.25))

print("skew: "+ str(skew))
sns.pairplot(df)
df.corr()
#Turn Up/Down to 0,1

df['DirectionNumber'] = df['Direction'].map({'Up': 1, 'Down': 0})

df.head()
#Create Input

inputDf = df[['Year', 'Lag1', 'Lag2', 'Lag3', 'Lag4', 'Lag5', 'Volume', 'Today']]

outputDf = df[['DirectionNumber']]
#Create Model

logisticRegr = LogisticRegression()

logisticRegr.fit(inputDf, outputDf)

print(logisticRegr.intercept_, logisticRegr.coef_)
#Prediction

y_pred = logisticRegr.predict(inputDf)



dresult = pd.DataFrame()

dresult["y_pred"] = y_pred;

dresult["y_original"] = df["DirectionNumber"];

print(dresult)
#counting result

correctValue =  len(dresult[(dresult['y_original'] == 1) & (dresult['y_pred'] == 1)])

incorrectValue = len(dresult[(dresult['y_original'] == 1) & (dresult['y_pred'] == 0)])

print("correct:" + str(correctValue))

print("incorrect:" + str(incorrectValue))

print("percentage:" + str(correctValue*100/(correctValue+incorrectValue)))
#counting result

correctValue =  len(dresult[(dresult['y_original'] == 0) & (dresult['y_pred'] == 0)])

incorrectValue = len(dresult[(dresult['y_original'] == 0) & (dresult['y_pred'] == 1)])

print("correct:" + str(correctValue))

print("incorrect:" + str(incorrectValue))

print("percentage:" + str(correctValue*100/(correctValue+incorrectValue)))
df.describe()
#Divide the data into Training and Test data sets

#Use data for 2005 as the test set and ALL other data as training set

dfWithOut2005 = df[(df['Year'] != 2005)]

inputDfWithOut2005 = dfWithOut2005[['Year', 'Lag1', 'Lag2', 'Lag3', 'Lag4', 'Lag5', 'Volume', 'Today']]

outputDfWithOut2005 = dfWithOut2005[['DirectionNumber']]

inputDfWithOut2005.describe()

#Create Model

logisticRegrTestTrain = LogisticRegression()

logisticRegrTestTrain.fit(inputDfWithOut2005, outputDfWithOut2005)

print(logisticRegrTestTrain.intercept_, logisticRegrTestTrain.coef_)
#Create Testing Data

dfOnly2005 = df[(df['Year'] == 2005)]

inputDfOnly2005 = dfOnly2005[['Year', 'Lag1', 'Lag2', 'Lag3', 'Lag4', 'Lag5', 'Volume', 'Today']]

outputDfOnly2005 = dfOnly2005['DirectionNumber']

outputDfOnly2005 = outputDfOnly2005.reset_index(drop=True)

inputDfOnly2005.describe()
#Prediction

y_pred_Only2005 = logisticRegrTestTrain.predict(inputDfOnly2005)



dresultTestTrain = pd.DataFrame()

dresultTestTrain["y_pred_Only2005"] = y_pred_Only2005;

dresultTestTrain["y_original_Only2005"] = outputDfOnly2005;

print(dresultTestTrain)
correctValue =  len(dresultTestTrain[(dresultTestTrain['y_original_Only2005'] == 1) & (dresultTestTrain['y_pred_Only2005'] == 1)])

incorrectValue = len(dresultTestTrain[(dresultTestTrain['y_original_Only2005'] == 1) & (dresultTestTrain['y_pred_Only2005'] != 1)])

print("correct:" + str(correctValue))

print("incorrect:" + str(incorrectValue))

print("percentage:" + str(correctValue*100/(correctValue+incorrectValue)))
correctValue =  len(dresultTestTrain[(dresultTestTrain['y_original_Only2005'] == 0) & (dresultTestTrain['y_pred_Only2005'] == 0)])

incorrectValue = len(dresultTestTrain[(dresultTestTrain['y_original_Only2005'] == 0) & (dresultTestTrain['y_pred_Only2005'] != 0)])

print("correct:" + str(correctValue))

print("incorrect:" + str(incorrectValue))

print("percentage:" + str(correctValue*100/(correctValue+incorrectValue)))
pFirstMethodCorrect = len(dresult[(dresult['y_original'] == 0) & (dresult['y_pred'] == 0)]) + len(dresult[(dresult['y_original'] == 1) & (dresult['y_pred'] == 1)])

pFirstMethodTotal = dresult['y_original'].count()

print('percentage Corrected of First Method:' + str(pFirstMethodCorrect*100/pFirstMethodTotal))
pSecondMethodCorrect = len(dresultTestTrain[(dresultTestTrain['y_original_Only2005'] == 0) & (dresultTestTrain['y_pred_Only2005'] == 0)]) + len(dresultTestTrain[(dresultTestTrain['y_original_Only2005'] == 1) & (dresultTestTrain['y_pred_Only2005'] == 1)])

pSecondMethodTotal = dresultTestTrain['y_original_Only2005'].count()

print('percentage Corrected of Second Method:' + str(pSecondMethodCorrect*100/pSecondMethodTotal))