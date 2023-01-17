import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

import pandas as pd

from sklearn.preprocessing import LabelEncoder
train = pd.read_csv('../input/covid19-global-forecasting-week-3/train.csv')
test = pd.read_csv('../input/covid19-global-forecasting-week-3/test.csv')
sub = pd.read_csv('../input/covid19-global-forecasting-week-3/submission.csv')
train.isnull().sum()
test.isnull().sum()
data = pd.concat([train, test])
data['Province_State']=data['Province_State'].fillna('PS',inplace=True)
train_len=len(train)
for cols in data.columns:

    if (data[cols].dtype==np.number):

        continue

    data[cols]=LabelEncoder().fit_transform(data[cols])
train=data[:train_len]
train = train.drop('ForecastId',axis=1)
test=data[train_len:]
drop=['Id','ConfirmedCases','Fatalities']

test = test.drop(drop,axis=1)
x_train=train.drop(labels=['Fatalities','ConfirmedCases','Id'],axis=1)

y_train1=train['ConfirmedCases']

y_train2=train['Fatalities']
from xgboost import XGBRegressor
x_test=test.drop(labels=['ForecastId'],axis=1)
xmodel1 = XGBRegressor(n_estimators=1000)

xmodel1.fit(x_train, y_train1)

y1_xpred = xmodel1.predict(x_test)
xmodel2 = XGBRegressor(n_estimators=1000)

xmodel2.fit(x_train, y_train2)

y2_xpred = xmodel2.predict(x_test)
data_to_submit = pd.DataFrame({

    'ForecastId':sub['ForecastId'],

    'ConfirmedCases':y1_xpred,

    'Fatalities':y2_xpred

})

data_to_submit.to_csv('submission.csv', index = False)
sub1 = pd.read_csv('submission.csv')
sub1.head()