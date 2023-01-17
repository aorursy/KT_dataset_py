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
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.ensemble import RandomForestRegressor
train = pd.read_csv('../input/covid19-global-forecasting-week-5/train.csv',parse_dates=['Date'])

test = pd.read_csv('../input/covid19-global-forecasting-week-5/test.csv',parse_dates=['Date'])

submission = pd.read_csv('../input/covid19-global-forecasting-week-5/submission.csv')



train['dayofyear'] = train.Date.dt.dayofyear

test['dayofyear'] = test.Date.dt.dayofyear
data = pd.concat([train,test],axis=0,sort=False)
data.sample(5)
(data['County'].isna() & data['Province_State'].isna()).sum()
data.loc[data['County'].isna() & data['Province_State'].isna(),'place_id'] = data.loc[data['County'].isna() & data['Province_State'].isna(),'Country_Region']

data.loc[data['County'].isna() & ~data['Province_State'].isna(),'place_id'] = data.loc[data['County'].isna() & ~data['Province_State'].isna(),'Country_Region'] + ' - ' + data.loc[data['County'].isna() & ~data['Province_State'].isna(),'Province_State']

data.loc[data['place_id'].isna(),'place_id'] = data.loc[data['place_id'].isna(),'Country_Region'] + ' - ' + data.loc[data['place_id'].isna(),'Province_State'] + ' - ' + data.loc[data['place_id'].isna(),'County']
data['place_id'].isna().sum()
data.drop(['County','Province_State','Country_Region','Date'],axis=1,inplace=True)
le = LabelEncoder()

data['place_id'] = le.fit_transform(data['place_id'])

data['Target'] = le.fit_transform(data['Target'])
train = data[~data.Id.isna()].drop(['ForecastId'],axis=1)

test = data[data.Id.isna()].drop(['Id'],axis=1)
test.set_index('ForecastId',inplace=True)

test.drop('TargetValue',axis=1,inplace=True)
test.sample(5)
def eval_score(true,pred,weight):

    true = np.array(true) if type(true) is pd.Series else true

    pred = np.array(pred) if type(pred) is pd.Series else pred

    weight = np.array(weight) if type(weight) is pd.Series else weight

    score = np.round((np.abs(true) * weight).sum()/len(true),4)

    return score
scores = []

result = pd.DataFrame(index=[],columns=['ForecastId','TargetValue'])

for SEED in range(100):

#     if SEED == 3:

#         break

    X = train.drop(['Id','TargetValue'],axis=1)

    y = train['TargetValue']

    X_train,X_valid,y_train,y_valid = train_test_split(X,y,random_state=SEED)

    model = RandomForestRegressor(random_state=SEED)

    model.set_params(n_estimators=10)

    model.fit(X_train,y_train)

#     score = model.score(X_valid,y_valid)

#     print('score: ' + str(round(score,4)))

    y_pred = model.predict(test)

    scores.append(eval_score(y_valid, y_pred, X_valid['Weight']))

    pred = y_pred.astype(int)

    pred[pred<0] = 0

    tmp = test.reset_index()[['ForecastId']]

    tmp['TargetValue'] = pred

    result = pd.concat([result,tmp])

result = result.astype(int)

print(np.mean(scores))

print(result.shape)

result.tail()
tmp = result.groupby('ForecastId')['TargetValue'].quantile(q=0.05)

tmp.rename('_0.05',inplace=True)

tmp = pd.DataFrame(tmp)

tmp['_0.5'] = result.groupby('ForecastId')['TargetValue'].quantile(q=0.5)

tmp['_0.95'] = result.groupby('ForecastId')['TargetValue'].quantile(q=0.95)

tmp.reset_index(inplace=True)

sub = pd.melt(tmp, id_vars=['ForecastId'], value_vars=['_0.05','_0.5','_0.95'])

sub['ForecastId_Quantile']=sub['ForecastId'].astype(str)+sub['variable']

sub = pd.merge(submission,sub,on='ForecastId_Quantile')

sub.drop('TargetValue',axis=1,inplace=True)

sub.rename(columns={'value':'TargetValue'},inplace=True)

sub=sub[['ForecastId_Quantile','TargetValue']]

sub.reset_index(drop=True,inplace=True)

# sub

sub.to_csv("submission.csv",index=False)