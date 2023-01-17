# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train=pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv")

test=pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/test.csv")

train.head()
test.head()
train.columns
train.corr()
import numpy as np

from scipy.stats import entropy

from math import log, e



""" Usage: pandas_entropy(df['column1']) """



def pandas_entropy(column, base=None):

    vc = pd.Series(column).value_counts(normalize=True, sort=False)

    base = e if base is None else base

    return -(vc * np.log(vc)/np.log(base)).sum()
for colnames in train.columns:

    entro=pandas_entropy(colnames)

    #print(colnames+" :"+str(entro))
hepsi=train.isnull().sum()

hepsi[0:20]
train.shape
del train["Alley"]
hepsi[20:40]
hepsi[40:60]
del train["FireplaceQu"]
hepsi[60:80]
del train["PoolQC"]

del train["Fence"]

del train["MiscFeature"]
test.isnull().sum()
train.shape
train.isnull().sum().sum()
train_numerical=train[train._get_numeric_data().columns]
train_numerical.shape
train_numerical.isnull().sum()
train_numerical["LotFrontage"].describe()
from sklearn.impute import SimpleImputer

imp = SimpleImputer(missing_values=np.nan, strategy='most_frequent')

latfromlange=imp.fit_transform(train_numerical)

latfromlange
latfromlange.shape
pedeler=pd.DataFrame(latfromlange)

pedeler.columns=train._get_numeric_data().columns

pedeler.head()
from sklearn.linear_model import LinearRegression

linear=LinearRegression()

x=pedeler.iloc[:,1:37]

y=pedeler.iloc[:,37:]
from sklearn.model_selection import cross_val_score

scores = cross_val_score(linear, x, y, cv=5)
scores
from sklearn.tree import DecisionTreeRegressor

dtr=DecisionTreeRegressor()

scores = cross_val_score(dtr, x, y, cv=5)

scores
import xgboost as xgb

xgreg=xgb.XGBRegressor()

scores = cross_val_score(dtr, x, y, cv=5)

scores
from sklearn.metrics import mean_squared_error

from sklearn.metrics import max_error

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)

xgreg=xgb.XGBRegressor()

xgreg.fit(x_train,y_train)

ypred=xgreg.predict(x_test)

mean_squared_error(y_pred=ypred,y_true=y_test)
from sklearn.metrics import max_error

max_error(y_pred=ypred,y_true=y_test)
from sklearn.metrics import mean_absolute_error

mean_absolute_error(y_pred=ypred,y_true=y_test)
from tpot import TPOTRegressor

tpot = TPOTRegressor(generations=1, population_size=50, verbosity=2)

tpot.fit(x_train,y_train)

print(tpot.score(x_test, y_test))

tpot.export('tpot_bir.py')
from sklearn.ensemble import RandomForestRegressor

rfr=RandomForestRegressor(bootstrap=True, max_features=0.7000000000000001, min_samples_leaf=1, min_samples_split=19, n_estimators=100)

rfr.fit(x_train,y_train)

ypred=rfr.predict(x_test)

print(mean_squared_error(y_pred=ypred,y_true=y_test))

print(max_error(y_pred=ypred,y_true=y_test))

print(mean_absolute_error(y_pred=ypred,y_true=y_test))
ypred=tpot.predict(x_test)

print(mean_squared_error(y_pred=ypred,y_true=y_test))

print(max_error(y_pred=ypred,y_true=y_test))

print(mean_absolute_error(y_pred=ypred,y_true=y_test))
test_ids=test.iloc[:,0:1]

test_ids


del test["Alley"]

del test["FireplaceQu"]

del test["PoolQC"]

del test["Fence"]

del test["MiscFeature"]
test_numerical=test[test._get_numeric_data().columns]
test_numerical["SalePrice"]=np.nan
latfromlangetest=imp.transform(test_numerical)

latfromlangetest
pedelertest=pd.DataFrame(latfromlangetest)

pedelertest=pedelertest.iloc[:,1:37]

pedelertest.head()
preds=tpot.predict(pedelertest)
pd.DataFrame(preds).shape
test_ids.shape
test_ids["SalePrice"]=preds
test_ids
test_ids.to_csv("tpotsubmin.csv",index=False)
from tpot import TPOTRegressor

tpot = TPOTRegressor(verbosity=2,max_time_mins=242)

tpot.fit(x_train,y_train)

print(tpot.score(x_test, y_test))

tpot.export('tpot_bir.py')
preds=tpot.predict(pedelertest)

test_ids["SalePrice"]=preds

test_ids.to_csv("tpotsubminsonson2.csv",index=False)