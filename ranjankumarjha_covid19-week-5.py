# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd

import numpy as np

from datetime import datetime

from sklearn import preprocessing

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor
df_train = pd.read_csv('../input/covid19-global-forecasting-week-5/train.csv')

df_test = pd.read_csv('../input/covid19-global-forecasting-week-5/test.csv')

sample = pd.read_csv('../input/covid19-global-forecasting-week-5/submission.csv')
df_train.sort_values(by=['TargetValue'])
df_train['Date'] = pd.to_datetime(df_train['Date'], infer_datetime_format=True)

df_test['Date'] = pd.to_datetime(df_test['Date'], infer_datetime_format=True)
df_train=df_train.drop(['Id','County','Province_State','Target'],axis=1)
df_test=df_test.drop(['ForecastId','County','Province_State','Target'],axis=1)
df_train.loc[:,'Date'] = df_train.Date.dt.strftime("%m%d")

df_train["Date"]  = df_train["Date"].astype(int)
df_test.loc[:,'Date'] = df_test.Date.dt.strftime("%m%d")

df_test["Date"]  = df_test["Date"].astype(int)
le = preprocessing.LabelEncoder()



df_train['Country_Region'] = le.fit_transform(df_train['Country_Region'])

df_test['Country_Region'] = le.fit_transform(df_test['Country_Region'])
predictors = df_train.drop(['TargetValue'], axis=1)

target = df_train["TargetValue"]

X_train, X_test, y_train, y_test = train_test_split(predictors, target, test_size = 0.22, random_state = 0)
model = RandomForestRegressor(n_jobs=-1)

estimators = 100

scores = []

model.set_params(n_estimators=estimators)

model.fit(X_train, y_train)

scores.append(model.score(X_test, y_test))
scores
y_pred2 = model.predict(X_test)

y_pred2
df_test.index.name = 'ForcastId'
predictions = model.predict(df_test)



pred_list = [int(x) for x in predictions]



df_out = pd.DataFrame({'ForcastId': df_test.index, 'TargetValue': pred_list})

print(df_out)
x=df_out.groupby(['ForcastId'])['TargetValue'].quantile(q=0.05).reset_index()

y=df_out.groupby(['ForcastId'])['TargetValue'].quantile(q=0.5).reset_index()

z=df_out.groupby(['ForcastId'])['TargetValue'].quantile(q=0.95).reset_index()
x.columns=['ForcastId','q0.05']

y.columns=['ForcastId','q0.5']

z.columns=['ForcastId','q0.95']

x=pd.concat([x,y['q0.5'],z['q0.95']],1)

x['q0.05']=x['q0.05'].clip(0,10000)

x['q0.5']=x['q0.5'].clip(0,10000)

x['q0.95']=x['q0.95'].clip(0,10000)
x['ForcastId'] =x['ForcastId']+ 1

x
sub=pd.melt(x, id_vars=['ForcastId'], value_vars=['q0.05','q0.5','q0.95'])

sub['variable']=sub['variable'].str.replace("q","", regex=False)

sub['ForecastId_Quantile']=sub['ForcastId'].astype(str)+'_'+sub['variable']

sub['TargetValue']=sub['value']

sub=sub[['ForecastId_Quantile','TargetValue']]

sub.reset_index(drop=True,inplace=True)

sub.to_csv("submission.csv",index=False)
sub.shape