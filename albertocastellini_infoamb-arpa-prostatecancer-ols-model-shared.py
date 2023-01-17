# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



# machine learning

from sklearn.linear_model import LinearRegression



# Any results you write to the current directory are saved as output.
# Data acquisition

data = pd.read_csv('../input/prostate.data',sep='\t')
data.shape

data.head(10)
data=data.drop(data.columns[0], axis=1)
data.head()
train=data['train']

type(train)

train.head()

data=data.drop('train',axis=1)

data.head()
lpsa=data['lpsa']

lpsa.head()

train
predictors=data.drop('lpsa',axis=1)

predictors.head()
data.head()
data.info()
data.hist(figsize=(20,15))

data.describe()
type(train)
dataTrain=data.loc[train=="T"]

dataTrain.head(10)
dataTrain.shape
dataTest=data.loc[train=="F"]

dataTest.head(10)
dataTest.shape
lpsaTrain=lpsa.loc[train=="T"]

lpsaTrain.head(10)
lpsaTrain.size
lpsaTest=lpsa.loc[train=="F"]

lpsaTest.head(10)
lpsaTest.size
dataTrain.corr()
predictorsTrain=dataTrain.drop('lpsa',axis=1)

predictorsTrain.head(10)
predictorsTrain.shape
predictorsTest=dataTest.drop('lpsa',axis=1)

predictorsTest.head(10)
predictorsTest.shape
predictorsTrainMeans=predictorsTrain.mean()

predictorsTrainStd=predictorsTrain.std()

print(predictorsTrainMeans)

print(predictorsTrainStd)

predictorsTrain_std=(predictorsTrain-predictorsTrainMeans)/predictorsTrainStd

predictorsTrain_std.head()
predictorsTrain_std.hist(figsize=(20,15))
predictorsTest_std=(predictorsTest-predictorsTrainMeans)/predictorsTrainStd

predictorsTest_std.head()
reg = LinearRegression()

reg.fit(predictorsTrain_std, lpsaTrain)

scoreTrain = reg.score(predictorsTrain_std, lpsaTrain)

scoreTrain
scoreTest = reg.score(predictorsTest_std, lpsaTest)

scoreTest
coeff_df = pd.DataFrame(predictorsTrain_std.columns)

coeff_df.columns = ['Feature']

coeff_df=pd.concat([pd.DataFrame([{'Feature': 'Intercept'}]), coeff_df], ignore_index=True)



myCoeff=pd.concat([pd.Series(reg.intercept_), pd.Series(reg.coef_)])

myCoeff.reset_index(drop=True,inplace=True)

coeff_df["Parameter"] = myCoeff

coeff_df



import statsmodels.api as sm

X2 = sm.add_constant(predictorsTrain_std)

est = sm.OLS(lpsaTrain, X2)

est2 = est.fit()

print(est2.summary())

#olsmod = sm.OLS(lpsaTrain, predictorsTrain_std)

#olsres = olsmod.fit()

#print(olsres.summary())
X2.head()
#est2.predict(predictorsTest_std)

Xnew = sm.add_constant(predictorsTest_std)

ynewpred =  est2.predict(Xnew) # predict out of sample

#print(ynewpred)



fig, ax = plt.subplots()

ax.plot(ynewpred, 'o', label="Pred")

ax.plot(lpsaTest, 'o', label="True")

ax.legend(loc="best");
#Mean prediction error on test data

meanPredErrTest=((ynewpred-lpsaTest).abs()).mean()

print(meanPredErrTest)
#Base error rate on test

baseErrOnTest=((lpsaTrain.mean()-lpsaTest).abs()).mean()

print(baseErrOnTest)