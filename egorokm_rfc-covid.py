# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt

import seaborn as sns

from datetime import datetime

import xgboost as xgb

from sklearn.ensemble import RandomForestClassifier



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# загружаем данные тренировочного набора



#data_train = pd.read_csv("data/train.csv")

data_train = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-1/train.csv")
# приводим признак дату к формату unix

data_train["Date"] = data_train["Date"].apply(lambda x: datetime.strptime(x,'%Y-%m-%d'))

data_train["Date"] = data_train["Date"].apply(lambda x: x.timestamp())

data_train["Date"]  = data_train["Date"].astype(int)
# выведем значение тренировочного набора данных

data_train.head()
# удалим столбец с городом и заполним пропущенные значения

data_train = data_train.drop(['Province/State'],axis=1)

data_train = data_train.dropna()

data_train.isnull().sum()
# загружаем тестовый набор данных

#data_test = pd.read_csv("data/test.csv")  

data_test = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-1/test.csv")   

data_test["Date"] = data_test["Date"].apply(lambda x: datetime.strptime(x,'%Y-%m-%d'))

data_test["Date"] = data_test["Date"].apply(lambda x: x.timestamp())

data_test["Date"] = data_test["Date"].astype(int)



data_test.isnull().sum()
data_test.drop('Province/State', axis = 1, inplace = True)

data_test.head()
data_test.info()
# выделяем признаки в модели

X = data_train[['Lat', 'Long', 'Date']]

# выделяем количество заболевших для предсказания

Y1 = data_train[['ConfirmedCases']]

# выделяем количество умерших для предсказания

Y2 = data_train[['Fatalities']]

X_test = data_test[['Lat', 'Long', 'Date']]
# определяем количество заболевших

model = RandomForestClassifier(bootstrap=True,max_depth=None, max_features='auto', max_leaf_nodes=None, 

                      n_estimators=150, random_state=None, n_jobs=1, verbose=0)

model.fit(X,Y1)

pred1 = model.predict(X_test)

pred1 = pd.DataFrame(pred1)

pred1.columns = ["ConfirmedCases_prediction"]
pred1.head()
# определяем количество умерших

model = RandomForestClassifier(bootstrap=True,max_depth=None, max_features='auto', max_leaf_nodes=None, 

                      n_estimators=150, random_state=None, n_jobs=1, verbose=0)

model.fit(X,Y2)

pred2 = model.predict(X_test)

pred2 = pd.DataFrame(pred2)

pred2.columns = ["Death_prediction"]
pred2.head()
data_submission = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-1/submission.csv")

data_submission.columns

sub_new = data_submission[["ForecastId"]]
concat = pd.concat([pred1,pred2,sub_new],axis=1)

concat.head()

concat.columns = ['ConfirmedCases', 'Fatalities', 'ForecastId']

concat = concat[['ForecastId','ConfirmedCases', 'Fatalities']]
concat["ConfirmedCases"] = concat["ConfirmedCases"].astype(int)

concat["Fatalities"] = concat["Fatalities"].astype(int)
concat.head()
concat.to_csv("submission.csv",index=False)