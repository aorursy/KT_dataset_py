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
import pandas as pd

import numpy as np 

import seaborn as sns

import matplotlib.pyplot as plt

import pandas as pd

from datetime import datetime

from sklearn import preprocessing

from xgboost import XGBRegressor

from sklearn.tree import DecisionTreeRegressor

from lightgbm import LGBMRegressor

from catboost import CatBoostRegressor

from sklearn.linear_model import LogisticRegression

from sklearn.linear_model import LinearRegression

from sklearn.ensemble import RandomForestRegressor 
train    = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/train.csv')

test     = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/test.csv')

submission = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/submission.csv')
#getting the first 5 rows

train.head()
#getting the first 5 rows of test data-set

test.head()
xtrain=train[['Id']]

test['Id']=test['ForecastId']

xtest=test[['Id']]

ytrain_cc=train[['ConfirmedCases']]

ytrain_ft=train[['Fatalities']]
xtr=np.array_split(xtrain,313)

ycc=np.array_split(ytrain_cc,313)

yft=np.array_split(ytrain_ft,313)

xte=np.array_split(xtest,313)
a=np.max(xtr[0]).values

b=a-71

b=b[0]
xte[0]=xte[0]+a

for i in range (312):

    xte[i+1]=xte[0]  
for i in range (312):

    xtr[i+1]=xtr[0]
from sklearn.linear_model import Lasso

from sklearn.preprocessing import PolynomialFeatures

poly=PolynomialFeatures(3)

y_pred_cc=[]

for i in range (313): #for loop is used to iterate through different regions

    xtr[i]=poly.fit_transform(xtr[i])

    xte[i]=poly.fit_transform(xte[i])

    model=Lasso()

    model.fit(xtr[i],ycc[i]);

    y_pr_cc=model.predict(xte[i])

    ycc[i]= ycc[i][71:]

    y_pr_cc=y_pr_cc[b:]

    y_pr_cc=np.append(ycc[i], y_pr_cc)

    y_pred_cc.append(y_pr_cc)
y_pred_ft=[]

for i in range (313): #for loop is used to iterate through different regions

    model=Lasso()

    model.fit(xtr[i],yft[i]);

    y_pr_ft=model.predict(xte[i])

    yft[i]= yft[i][71:]

    y_pr_ft=y_pr_ft[b:]

    y_pr_ft=np.append(yft[i], y_pr_ft)

    y_pred_ft.append(y_pr_ft);
y_pred_ft_1 = [item for sublist in y_pred_ft for item in sublist]

y_pred_cc_1 = [item for sublist in y_pred_cc for item in sublist]
df_final=pd.DataFrame({'ForecastId':submission.ForecastId, 'ConfirmedCases':y_pred_cc_1, 'Fatalities':y_pred_ft_1})

df_final.to_csv('/kaggle/working/submission.csv', index=False)

data=pd.read_csv('/kaggle/working/submission.csv')

data.head(20)