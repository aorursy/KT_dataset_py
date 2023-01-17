import pandas as pd 

import datetime

import numpy as np 

import matplotlib.pyplot as plt 



from sklearn.linear_model import LinearRegression

from sklearn.model_selection import TimeSeriesSplit

from tqdm.notebook import tqdm
sputniks = pd.read_csv('/kaggle/input/sputnik/train.csv')

sputniks.rename(columns={'epoch':'Datetime'}, inplace=True)

sputniks['Datetime'] = pd.to_datetime(sputniks.Datetime)

sputniks.index  = sputniks.Datetime

sputniks.drop('Datetime', axis = 1, inplace = True)

sputniks.head(2)
sputniks['day'] = sputniks.index.day

sputniks['hour'] = sputniks.index.hour

sputniks['minute'] = sputniks.index.minute

sputniks['seconds'] = sputniks.index.second

sputniks['dayofweek'] = sputniks.index.dayofweek

sputniks['quarter'] = sputniks.index.quarter

sputniks['dayofyear'] = sputniks.index.dayofyear

sputniks['dayofmonth'] = sputniks.index.day

sputniks['weekofyear'] = sputniks.index.weekofyear

sputniks['error']  = np.linalg.norm(sputniks[['x', 'y', 'z']].values - sputniks[['x_sim', 'y_sim', 'z_sim']].values, axis=1)

sputniks.head(2)
res=pd.DataFrame(['id','error'])

for i in tqdm(range(600)):

    x_train = sputniks[(sputniks.sat_id == i)&(sputniks.type == 'train')].copy()

    x_test = sputniks[(sputniks.sat_id == i)&(sputniks.type == 'test')].copy()



    x_train.drop(['sat_id', 'id', 'type', 'x', 'y', 'z'], axis = 1, inplace = True)

    

    x_test.drop(['sat_id', 'type', 'x', 'y', 'z'], axis = 1, inplace = True)

    model = LinearRegression()

    model.fit(np.asarray(x_train.drop(['error'], axis = 1)),np.asarray(x_train['error']))

    forecast = model.predict(np.asarray(x_test.drop(['id','error'],axis=1)))

    print(forecast.shape)

    #print(x_test.loc[:,'error'].shape)

    x_test['error'] = forecast

    res = pd.concat([res, x_test[['id','error']]])
res
res.index=[i for i in range(res.shape[0])]
result=res.iloc[2:][['id','error']]
result.astype({'id': 'int32'}).to_csv('pred.csv', index=False)