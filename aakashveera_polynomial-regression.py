import numpy as np # linear algebra

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt
train = pd.read_csv("../input/covid19-global-forecasting-week-4/train.csv")



test = pd.read_csv("../input/covid19-global-forecasting-week-4/test.csv")
train['Date'] = pd.to_datetime(train['Date'])

test['Date'] = pd.to_datetime(test['Date'])
train = train[train['Date']<='2020-04-14']

train['Location'] = train['Province_State'].astype(str) + train['Country_Region'].astype(str)

test['Location'] = test['Province_State'].astype(str) + test['Country_Region'].astype(str)

test = test.merge(train[['ConfirmedCases','Fatalities','Location','Date']],how='left',on=['Location','Date'])

train = train[train['Date']<='2020-04-01']

data = pd.concat([train,test],axis=0)

data = data.sort_values(['Country_Region','Date'])
data.drop(['Id','Province_State','Country_Region'],axis=1,inplace=True)
import gc

del train,test

gc.collect()
data
locations = data['Location'].unique()
from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree=4)

from sklearn.linear_model import LinearRegression

from tqdm import tqdm
Res = None



for i in tqdm(locations):

    df = data[data['Location']==i].reset_index().drop(['index'],axis=1).reset_index() 

    X_c = poly.fit_transform(np.array(df['index']).reshape(-1,1))

    df[[0,1,2,3,4]] = pd.DataFrame(X_c)

    

    train = df[df['Date']<='2020-04-14'][['ForecastId','ConfirmedCases','Fatalities',0,1,2,3,4]]

    test = df[df['Date']>='2020-04-15'][['ForecastId','ConfirmedCases','Fatalities',0,1,2,3,4]]

    

    

    model = LinearRegression()

    model.fit(train.drop(['ConfirmedCases','ForecastId','Fatalities'],axis=1),train['ConfirmedCases'])

    test['ConfirmedCases']=model.predict(test.drop(['ConfirmedCases','ForecastId','Fatalities'],axis=1))

    

    model = LinearRegression()

    model.fit(train.drop(['ConfirmedCases','ForecastId','Fatalities'],axis=1),train['Fatalities'])

    test['Fatalities']=model.predict(test.drop(['ConfirmedCases','ForecastId','Fatalities'],axis=1))

    

    pred = pd.concat([train[-13:][['ForecastId','ConfirmedCases','Fatalities']],test[['ForecastId','ConfirmedCases','Fatalities']]])

    

    Res = pd.concat([Res,pred],axis=0)
Res['ForecastId'] = Res['ForecastId'].astype(int)
Res.to_csv("submission.csv",index=False)
Res