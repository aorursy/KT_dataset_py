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
d = pd.read_csv("../input/cats.csv")

d.head()
print("Check for NaN/null values:\n",d.isnull().values.any())

print("Number of NaN/null values:\n",d.isnull().sum())
main = sm.ols(formula="Hwt ~ Bwt+Sex",data=d).fit()

print(main.summary())
main_i = sm.ols(formula="Hwt ~ Bwt*Sex",data=d).fit()

print(main_i.summary())
m = main.predict(pd.DataFrame([['F',3.5]],columns = ['Sex', 'Bwt']))

m_i = main_i.predict(pd.DataFrame([['F',3.5]],columns = ['Sex', 'Bwt']))

print("Main-effect only model prediction:\n",m)

print("Interaction model prediction:\n",m_i)
db = pd.read_csv("../input/trees.csv")

db.head()
print("Check for NaN/null values:\n",db.isnull().values.any())

print("Number of NaN/null values:\n",db.isnull().sum())
main_b = sm.ols(formula="Volume ~ Girth+Height",data=db).fit()

print(main_b.summary())
main_b_i = sm.ols(formula="Volume ~ Girth*Height",data=db).fit()

print(main_b_i.summary())
main_b_log = sm.ols(formula="Volume ~ np.log1p(Girth)+np.log1p(Height)",data=db).fit()

print(main_b_log.summary())
main_b_i_log = sm.ols(formula="Volume ~ np.log1p(Girth)*np.log1p(Height)",data=db).fit()

print(main_b_i_log.summary())
dc = pd.read_csv("../input/mtcars.csv")

dc.head()
print("Check for NaN/null values:\n",dc.isnull().values.any())

print("Number of NaN/null values:\n",dc.isnull().sum())
main_c = sm.ols(formula="mpg ~ wt+hp*C(cyl)",data=dc).fit()

print(main_c.summary())
test = pd.DataFrame([[4,100,2.100],[8,210,3.900],[6,200,2.900]],columns = ['cyl','hp','wt'])

MPG = main_c.predict(test)

MPG
dd = pd.read_csv("../input/diabetes.csv")

dd.head()
print("Check for NaN/null values:\n",dd.isnull().values.any())

print("Number of NaN/null values:\n",dd.isnull().sum())
dd = dd.dropna()
print("Check for NaN/null values:\n",dd.isnull().values.any())

print("Number of NaN/null values:\n",dd.isnull().sum())
diaNull = sm.ols(formula="chol ~ 1",data=dd).fit()

print(diaNull.summary())
diaFull = sm.ols(formula="chol ~ age*gender*weight*frame+waist*height*hip+location",data=dd).fit()

print(diaFull.summary())
from mlxtend.feature_selection import SequentialFeatureSelector as sfs
df = pd.get_dummies(dd, prefix=['gender','frame','location'], columns=['gender','frame','location'])

df.head()
inputDF = df[["age","gender_female","gender_male","frame_large","frame_medium","frame_small","weight","waist","height","hip","location_Buckingham","location_Louisa"]]

outputDF = df[["chol"]]



model = sfs(LinearRegression(),k_features=5,forward=True,verbose=2,cv=5,n_jobs=-1,scoring='r2')

model.fit(inputDF,outputDF)
model.k_feature_names_
backwardModel = sfs(LinearRegression(),k_features=5,forward=False,verbose=2,cv=5,n_jobs=-1,scoring='r2')

backwardModel.fit(inputDF,outputDF)
backwardModel.k_feature_names_