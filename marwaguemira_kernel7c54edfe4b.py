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
import warnings

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import MinMaxScaler

from xgboost import XGBRegressor

warnings.filterwarnings('ignore')
df_train=pd.read_csv("/kaggle/input/covid19-global-forecasting-week-4/train.csv")

df_train.tail(5)
df_train.isna().sum()
df_train['Province_State'].isna().sum()
df_train['Province_State']=df_train['Province_State'].fillna('')

df_train['Province_State'].isna().sum()
df_train['Country_Region']=df_train['Country_Region']+'_'+df_train['Province_State']
#df_train.sort_values(by=['Date'],inplace=True)
df_train
df_test=pd.read_csv("/kaggle/input/covid19-global-forecasting-week-4/test.csv")

df_test.sample(5)
df_test.isna().sum()
df_test['Province_State']=df_test['Province_State'].fillna('')

df_test['Province_State'].isna().sum()
df_test
df_test['Country_Region']=df_test['Country_Region']+'_'+df_test['Province_State']
df_test.sample(8)
def create_features(df,label=None):

    """

    Creates time series features from datetime index.

    """

    df = df.copy()

    df['Date'] = df.index

    df['dayofweek'] = df['Date'].dt.dayofweek

    df['quarter'] = df['Date'].dt.quarter

    df['month'] = df['Date'].dt.month

    

    df['dayofyear'] = df['Date'].dt.dayofyear

    df['dayofmonth'] = df['Date'].dt.day

    df['weekofyear'] = df['Date'].dt.weekofyear

    

    X = df[['dayofweek','quarter','month',

           'dayofyear','dayofmonth','weekofyear']]

   

    return X
train = df_train.set_index(['Date'])

train.index = pd.to_datetime(train.index)

train_features=pd.DataFrame(create_features(train))

train_features
df_train_final = pd.concat([train,train_features], axis=1)

df_train_final.reset_index(drop=True ,inplace =True )

df_train_final
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

df_train_final['Country_Region'] = le.fit_transform(df_train_final['Country_Region'])
df_train_final.sample(8)
df_train_final.drop(columns='Province_State' , inplace=True)

df_train_final
test = df_test.set_index(['Date'])

test.index = pd.to_datetime(test.index)

test_features=pd.DataFrame(create_features(test))

test_features
df_test_final = pd.concat([test,test_features], axis=1)

df_test_final.reset_index(drop=True ,inplace =True )

df_test_final
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

df_test_final['Country_Region'] = le.fit_transform(df_test_final['Country_Region'])
df_test_final.sample(8)
df_test_final.drop(columns='Province_State' , inplace=True)

df_test_final
df_test_final = df_test_final.drop(['ForecastId'],axis=1)
df_train_final.corr()
import seaborn as sns

import matplotlib.pyplot as plt

plt.figure(figsize=(15,8))



sns.heatmap(df_train_final.corr(),annot=True , linewidth=1.5 )
df_train_final['ConfirmedCases'] = df_train_final['ConfirmedCases'].apply(int)

df_train_final['Fatalities'] = df_train_final['Fatalities'].apply(int)
df_train_final.columns
y_conf=df_train_final['ConfirmedCases']

y_conf.sample(6)

y_fat=df_train_final['Fatalities']
df_train_final_fat = df_train_final.drop(['Id','Fatalities'],axis=1)

df_train_final_fat.columns


df_train_final_ConCases = df_train_final.drop(['Id','ConfirmedCases','Fatalities'],axis=1)

df_train_final_ConCases.columns
df_train_final_fat
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(df_train_final_ConCases,y_conf,test_size=0.2,random_state=42)
rf = XGBRegressor(n_estimators = 2400 , random_state = 0 , max_depth = 26)

rf.fit(X_train,y_train)
pred_conf = rf.predict(X_test)

predictions = [round(value) for value in pred_conf]

predictions
rf.score(X_test,y_test)
from sklearn.metrics import mean_squared_error

r=np.sqrt(mean_squared_error(y_test,predictions))

r
from sklearn.metrics import mean_absolute_error

mae = mean_absolute_error(y_test,predictions)

print('MAE: %f' % mae)
from sklearn.metrics import mean_squared_error

mse = mean_squared_error(y_test,predictions)

print('MSE: %f' % mse)
from math import sqrt

rmse = sqrt(mse)

print('RMSE: %f' % rmse)
from sklearn.metrics import r2_score

R2=r2_score(y_test,predictions)

print('R2: %f' % R2)
dfpred = pd.DataFrame(data=predictions, columns=[ "pred"] , index=y_test.index )

dfpred1=pd.concat([dfpred,y_test],axis=1)

dfpred1.sample(30)
from sklearn.model_selection import train_test_split

X_train2,X_test2,y_train2,y_test2 = train_test_split(df_train_final_fat,y_fat,test_size=0.2,random_state=42)
X_train2
rf2 = XGBRegressor(n_estimators = 2000 , random_state = 0 , max_depth = 20)

rf2.fit(X_train2,y_train2)
pred_fat = rf2.predict(X_test2)

predictions_fat = [round(value) for value in pred_fat]

predictions_fat
rf2.score(X_test2,y_test2)
from sklearn.metrics import mean_squared_error

r=np.sqrt(mean_squared_error(y_test2,predictions_fat))

r
from sklearn.metrics import mean_absolute_error

mae2 = mean_absolute_error(y_test2,predictions_fat)

print('MAE: %f' % mae2)
from sklearn.metrics import mean_squared_error

mse2 = mean_squared_error(y_test2,predictions_fat)

print('MSE: %f' % mse2)
from math import sqrt

rmse2 = sqrt(mse2)

print('RMSE: %f' % rmse2)
from sklearn.metrics import r2_score

R2_fat=r2_score(y_test2,predictions_fat)

print('R2: %f' % R2_fat)
dfpred_fat = pd.DataFrame(data=predictions_fat, columns=[ "pred"] , index=y_test2.index )

dfpred_fat1=pd.concat([dfpred_fat,y_test2],axis=1)

dfpred_fat1.sample(30)
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test2,predictions_fat)

print("Accuracy: %.2f%%" % (accuracy * 100.0))
xgb = XGBRegressor(n_estimators = 2400 , random_state = 0 , max_depth = 26)

xgb.fit(df_train_final_ConCases,y_conf)
df_train_final_fat
pred_conf = xgb.predict(df_test_final)

predictions_conf = [round(value) for value in pred_conf]

predictions_conf
predictions_conf  = np.around(predictions_conf ,decimals = 0)

predictions_conf 
conf_test=pd.DataFrame(predictions_conf,index=df_test_final.index,columns=["ConfirmedCases"])

df_test_final=pd.concat([df_test_final,conf_test],axis=1)
df_test_final.loc[10135]
xgb2 = XGBRegressor(n_estimators = 2400 , random_state = 0 , max_depth = 26)

xgb2.fit(df_train_final_fat,y_fat)
df_test_final
df_test_final=df_test_final[['Country_Region', 'ConfirmedCases', 'dayofweek', 'quarter', 'month',

       'dayofyear', 'dayofmonth', 'weekofyear']]
df_test_final
pred_fat = xgb2.predict(df_test_final)

predictions_fat = [round(value) for value in pred_fat]

predictions_fat
predictions_fat  = np.around(predictions_fat ,decimals = 0)

predictions_fat 
df_sub=pd.read_csv("/kaggle/input/covid19-global-forecasting-week-4/submission.csv")

df_sub.tail(5)
df_sub['ConfirmedCases']=predictions_conf

df_sub['Fatalities']=predictions_fat
df_sub
df_sub.to_csv("submission.csv" , index = False)