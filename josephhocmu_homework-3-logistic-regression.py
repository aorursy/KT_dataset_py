!pip install regressors
import numpy as np 

import pandas as pd 

from regressors import stats

from sklearn import linear_model as lm

import statsmodels.formula.api as sm

import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression 

from sklearn.preprocessing import PolynomialFeatures

from sklearn.model_selection import train_test_split

from sklearn.metrics import r2_score

from sklearn.linear_model import LogisticRegression

from sklearn import metrics

from scipy.stats import skew

import seaborn as sns



import os

print(os.listdir("../input"))
d = pd.read_csv("../input/smarket.csv")

d.head(10)
print("Summary for column Year:\n", d["Year"].describe())
plt.boxplot(d["Year"])

plt.show()
skew(d["Year"])
print("Summary for column Lag1:\n", d["Lag1"].describe())
plt.boxplot(d["Lag1"])

plt.show()
skew(d["Lag1"])
print("Summary for column Lag2:\n", d["Lag2"].describe())
plt.boxplot(d["Lag2"])

plt.show()
skew(d["Lag2"])
print("Summary for column Lag3:\n", d["Lag3"].describe())
plt.boxplot(d["Lag3"])

plt.show()
skew(d["Lag3"])
print("Summary for column Lag4:\n", d["Lag4"].describe())
plt.boxplot(d["Lag4"])

plt.show()
skew(d["Lag4"])
print("Summary for column Lag5:\n", d["Lag5"].describe())
plt.boxplot(d["Lag5"])

plt.show()
skew(d["Lag5"])
print("Summary for column Volume:\n", d["Volume"].describe())
plt.boxplot(d["Volume"])

plt.show()
skew(d["Volume"])
print("Summary for column Today:\n", d["Today"].describe())
plt.boxplot(d["Today"])

plt.show()
skew(d["Today"])
d = d.drop([d.columns[0]],axis=1)
d.corr()
sns.pairplot(d)
d = pd.read_csv("../input/smarket.csv")

d = d.drop([d.columns[0]],axis=1)

d['Direction'] = d['Direction'].map({'Up': 1, 'Down': 0})

d.head(10)
#Model Fit 

inputDf = d[["Year", "Lag1", "Lag2", "Lag3", "Lag4", "Lag5", "Volume", "Today"]]

outputDf = d[["Direction"]].values.ravel()
logisticRegr = LogisticRegression(solver='lbfgs')

logisticRegr.fit(inputDf, outputDf)

print(logisticRegr.intercept_)

print(logisticRegr.coef_)
ynew = logisticRegr.predict(inputDf)
newDf = pd.DataFrame()

newDf["original"] = d["Direction"]

newDf["prediction"] = ynew

newDf.head()
i = 0

correctCount = 0

incorrectCount = 0



while i < len(newDf):

    if (newDf.at[i, 'original'] == 1) & (newDf.at[i,'prediction'] == 1):

        correctCount += 1

    elif (newDf.at[i, 'original'] == 1) & (newDf.at[i,'prediction'] == 0):

        incorrectCount += 1

    i += 1



print("Correct Count of Up is: ", correctCount)

print("Incorrect Count of Up is: ", incorrectCount)

print("Percent of Outcomes predicted correctly: ", (correctCount/(correctCount + incorrectCount)*100))
i = 0

correctCount = 0

incorrectCount = 0



while i < len(newDf):

    if (newDf.at[i, 'original'] == 0) & (newDf.at[i,'prediction'] == 0):

        correctCount += 1

    elif (newDf.at[i, 'original'] == 0) & (newDf.at[i,'prediction'] == 1):

        incorrectCount += 1

    i += 1



print("Correct Count of Down is: ", correctCount)

print("Incorrect Count of Down is: ", incorrectCount)

print("Percent of Outcomes predicted correctly: ", (correctCount/(correctCount + incorrectCount)*100))
d
dTrain = d[d.Year != 2005]

dTrain.tail()
#Model Fit 

inputDf = dTrain[["Year", "Lag1", "Lag2", "Lag3", "Lag4", "Lag5", "Volume", "Today"]]

outputDf = dTrain[["Direction"]].values.ravel()
logisticRegr2 = LogisticRegression(solver='lbfgs')

logisticRegr2.fit(inputDf, outputDf)

print(logisticRegr.intercept_)

print(logisticRegr.coef_)
dTest = d[d.Year == 2005]

dTest = dTest.reset_index(drop = True)

dTest.head()
inputDf = dTest[["Year", "Lag1", "Lag2", "Lag3", "Lag4", "Lag5", "Volume", "Today"]]

outputDf = dTest[["Direction"]].values.ravel()

ynew2 = logisticRegr2.predict(inputDf)
newDf2 = pd.DataFrame()

newDf2["original"] = outputDf

newDf2["prediction"] = ynew2

newDf2.head()
i = 0

correctCount = 0

incorrectCount = 0



while i < len(newDf2):

    if (newDf2.at[i, 'original'] == 1) & (newDf2.at[i,'prediction'] == 1):

        correctCount += 1

    elif (newDf2.at[i, 'original'] == 1) & (newDf2.at[i,'prediction'] == 0):

        incorrectCount += 1

    i += 1



print("Correct Count of Up is: ", correctCount)

print("Incorrect Count of Up is: ", incorrectCount)

print("Percent of Outcomes predicted correctly: ", (correctCount/(correctCount + incorrectCount)*100))
i = 0

correctCount = 0

incorrectCount = 0



while i < len(newDf2):

    if (newDf2.at[i, 'original'] == 0) & (newDf2.at[i,'prediction'] == 0):

        correctCount += 1

    elif (newDf2.at[i, 'original'] == 0) & (newDf2.at[i,'prediction'] == 1):

        incorrectCount += 1

    i += 1



print("Correct Count of Down is: ", correctCount)

print("Incorrect Count of Down is: ", incorrectCount)

print("Percent of Outcomes predicted correctly: ", (correctCount/(correctCount + incorrectCount)*100))