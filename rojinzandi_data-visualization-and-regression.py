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
import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn import preprocessing

import time

from datetime import datetime

from scipy import integrate, optimize

import warnings

warnings.filterwarnings('ignore')



# ML libraries

import lightgbm as lgb

from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

from sklearn import linear_model

from sklearn.metrics import mean_squared_error     
submission_example = pd.read_csv("../input/covid19-global-forecasting-week-3/submission.csv")

test = pd.read_csv("../input/covid19-global-forecasting-week-3/test.csv")

train = pd.read_csv("../input/covid19-global-forecasting-week-3/train.csv")

display(train.head(5))

display(test.head(5))



print("Train Data:")

print("Dates go from day", max(train['Date']), "to day", min(train['Date']), ", a total of", train['Date'].nunique(), "days")

print("Test Data:")

print("Dates go from day", max(test['Date']), "to day", min(test['Date']), ", a total of", test['Date'].nunique(), "days")

total_data = pd.concat([train,test],axis=0,sort=False) # join train and test

total_data.isna().sum() # verifying na

train.describe()
confirmed_total_date_Iran = train[train['Country_Region']=='Iran'].groupby(['Date']).agg({'ConfirmedCases':['sum']})

fatalities_total_date_Iran = train[train['Country_Region']=='Iran'].groupby(['Date']).agg({'Fatalities':['sum']})



plt.figure(figsize=(17,10))

plt.subplot(2, 2, 1)

confirmed_total_date_Iran.plot(ax=plt.gca(), title='Iran Confirmed')

plt.ylabel("Confirmed infection cases", size=13)



plt.subplot(2, 2, 2)

fatalities_total_date_Iran.plot(ax=plt.gca(), title='Iran Fatalities')

plt.ylabel("Fatalities cases", size=13)









confirmed_total_date_Italy = train[train['Country_Region']=='Italy'].groupby(['Date']).agg({'ConfirmedCases':['sum']})

fatalities_total_date_Italy = train[train['Country_Region']=='Italy'].groupby(['Date']).agg({'Fatalities':['sum']})



plt.figure(figsize=(17,10))

plt.subplot(2, 2, 1)

confirmed_total_date_Italy.plot(ax=plt.gca(), title='Italy Confirmed')

plt.ylabel("Confirmed infection cases", size=13)



plt.subplot(2, 2, 2)

fatalities_total_date_Italy.plot(ax=plt.gca(), title='Italy Fatalities')

plt.ylabel("Fatalities cases", size=13)