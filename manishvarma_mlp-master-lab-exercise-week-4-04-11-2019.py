!pip install regressors
import numpy as np 

import pandas as pd 

import os

import statsmodels.formula.api as sm

import statsmodels.sandbox.tools.cross_val as cross_val

from sklearn.preprocessing import PolynomialFeatures

from sklearn import linear_model as lm

from regressors import stats

from sklearn import metrics

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import KFold, cross_val_score,cross_val_predict, LeaveOneOut



print(os.listdir("../input"))
#Interactive Terms: Statsmodel

d = pd.read_csv("../input/diabetes.csv")

d.head()
main = sm.ols(formula="chol ~ age+frame",data=d).fit()

print(main.summary())
inter = sm.ols(formula="chol ~ age*frame",data=d).fit()

print(inter.summary())
inter = sm.ols(formula="chol ~ gender*frame",data=d).fit()

print(inter.summary())
inter = sm.ols(formula="chol ~ height*weight",data=d).fit()

print(inter.summary())
import statsmodels.api as sma

d = pd.read_csv("../input/diabetes.csv")

d.head()
chol1 = sm.ols(formula="chol ~ 1",data=d).fit()

chol2 = sm.ols(formula="chol ~ age",data=d).fit()

chol3 = sm.ols(formula="chol ~ age+frame",data=d).fit()

chol4 = sm.ols(formula="chol ~ age*frame",data=d).fit()
print(sma.stats.anova_lm(chol1,chol2,chol3,chol4))
from mlxtend.feature_selection import SequentialFeatureSelector as sfs
d = pd.read_csv("../input/nuclear.csv")

d = d.rename(index=str,columns={"cum.n":"cumn"})

d.head()
df = pd.read_csv("../input/nuclear.csv")

df = df.rename(index=str,columns={"cum.n":"cumn"})

inputDF = df[["date","cap","pt","t1","t2","pr","ne","ct","bw"]]

outputDF = df[["cost"]]



model = sfs(LinearRegression(),k_features=5,forward=True,verbose=2,cv=5,n_jobs=-1,scoring='r2')

model.fit(inputDF,outputDF)
#Selected feature index.

model.k_feature_idx_
#Column names for the selected feature.

model.k_feature_names_
# Backward Selection: Scikit-Learn 

df = pd.read_csv("../input/nuclear.csv")

df = df.rename(index=str,columns={"cum.n":"cumn"})

inputDF = df[["date","cap","pt","t1","t2","pr","ne","ct","bw"]]

outputDF = df[["cost"]]



backwardModel = sfs(LinearRegression(),k_features=5,forward=False,verbose=2,cv=5,n_jobs=-1,scoring='r2')

backwardModel.fit(inputDF,outputDF)
#Selected feature index.

backwardModel.k_feature_idx_
#Column name for the selected feature.

backwardModel.k_feature_names_
from sklearn import metrics

from sklearn.linear_model import LinearRegression
d=pd.read_csv("../input/auto.csv")

d.head()
#LOOCV: Scikit-Learn 

df = pd.read_csv("../input/auto.csv")

df = df.drop(columns=["name"])

df.head()
inputDF = df[["mpg"]]

outputDF = df[["horsepower"]]

model = LinearRegression()

loocv = LeaveOneOut()



rmse = np.sqrt(-cross_val_score(model, inputDF, outputDF, scoring="neg_mean_squared_error", cv = loocv))

print(rmse.mean())
predictions = cross_val_predict(model, inputDF, outputDF, cv=loocv)
df = pd.read_csv("../input/auto.csv")

df = df.drop(columns=["name"])

df.head()
#kFCV: Scikit-Learn

inputDF = df[["mpg"]]

outputDF = df[["horsepower"]]

model = LinearRegression()

kf = KFold(5, shuffle=True, random_state=42).get_n_splits(inputDF)

rmse = np.sqrt(-cross_val_score(model, inputDF, outputDF, scoring="neg_mean_squared_error", cv = kf))

print(rmse.mean())
predictions = cross_val_predict(model, inputDF, outputDF, cv=kf)