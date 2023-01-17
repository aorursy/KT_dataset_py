!pip install plotly
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np 

import pandas as pd

import seaborn as sns

from scipy import stats

import matplotlib.pyplot as plt

import plotly.express as px

from datetime import datetime

%matplotlib inline



from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import GridSearchCV, KFold



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train = pd.read_csv('../input/covid19-global-forecasting-week-5/train.csv')

test = pd.read_csv('../input/covid19-global-forecasting-week-5/test.csv')

submission = pd.read_csv('../input/covid19-global-forecasting-week-5/submission.csv')
train.columns
train.isnull().sum()
test.isnull().sum()
submission
submission.shape
submission['TargetValue'].sum()
# 'TargetValue'에 따라 오름차순으로 정렬

train.sort_values(by=['TargetValue'])
fig = px.pie(train, values = 'TargetValue', names='Target')

fig.update_traces(textposition = 'inside')

fig.update_layout(uniformtext_minsize=12, uniformtext_mode='hide')

fig.show()
fig = px.pie(train, values='TargetValue', names='Country_Region')

fig.update_traces(textposition='inside')

fig.update_layout(uniformtext_minsize=12, uniformtext_mode='hide')

fig.show()
getToplist = 15

grouped_multiple = train.groupby(['Country_Region'], as_index=False)['TargetValue'].sum()

countryTop = grouped_multiple.nlargest(getToplist, 'TargetValue')['Country_Region']

newlist = train[train['Country_Region'].isin(countryTop.values)]

line = newlist.groupby(['Date', 'Country_Region'], as_index=False)['TargetValue'].sum()

line = line[line['TargetValue'] >= 0]
line.pivot(index='Date', columns='Country_Region', values='TargetValue').plot(figsize=(10,5))

plt.grid(zorder=0)

plt.title('Top' + str(getToplist) + 'ConfirmedCases & Fatalities', fontsize=18, pad=10)

plt.ylabel('People')

plt.xlabel('Date')

plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))

plt.show()
train = train.drop(['County', 'Province_State','Country_Region','Target'], axis=1)

test = test.drop(['County', 'Province_State','Country_Region','Target'], axis=1)



train.head()
from sklearn.preprocessing import OrdinalEncoder



def create_feature(df):

    df['day'] = df['Date'].dt.day

    df['month'] = df['Date'].dt.month

    df['dayofweek'] = df['Date'].dt.dayofweek

    df['dayofyear'] = df['Date'].dt.dayofyear

    df['quarter'] = df['Date'].dt.quarter

    df['weekofyear'] = df['Date'].dt.weekofyear

    return df



# Date 변수 분리
def train_dev_split(df, days):

    date = df['Date'].max() - dt.timedelta(days=days)

    return df[df['Date'] <= date], df[df['Date'] > date]
test_date_min = test['Date'].min()

test_date_max = test['Date'].max()
def avoid_date_leakage(df, date=test_date_min):

    return df[df['Date'] < date]
def to_integer(dt_time):

    return 10000*dt_time.year + 100*dt_time.month + dt_time.day
train['Date'] = pd.to_datetime(train['Date'])

test['Date'] = pd.to_datetime(test['Date'])
train['Date'] = train['Date'].dt.strftime('%Y%m%d')

test['Date'] = test['Date'].dt.strftime('%Y%m%d')
train.head()
from sklearn.model_selection import train_test_split



predictors = train.drop(['TargetValue', 'Id'], axis=1)

target = train['TargetValue']

X_train, X_test, y_train, y_test = train_test_split(predictors, target, test_size=0.22, random_state=0)
model = RandomForestRegressor(n_jobs=-1)

estimators=100

scores=[]

model.set_params(n_estimators=estimators)

model.fit(X_train, y_train)

scores.append(model.score(X_test, y_test))
X_test
test.drop(['ForecastId'], axis=1, inplace=True)

test.index.name = 'Id'

test
y_pred2 = model.predict(X_test)

y_pred2
predictions = model.predict(test)



pred_list = [int(x) for x in predictions]



output = pd.DataFrame({'Id': test.index, 'TargetValue': pred_list})

print(output)
output
a = output.groupby(['Id'])['TargetValue'].quantile(q=0.05).reset_index() # 5% 지점

b = output.groupby(['Id'])['TargetValue'].quantile(q=0.5).reset_index() # 절반 지점

c = output.groupby(['Id'])['TargetValue'].quantile(q=0.05).reset_index() # 95% 지점
a.columns = ['Id', 'q0.05']

b.columns = ['Id', 'q0.5']

c.columns = ['Id', 'q0.95']



a = pd.concat([a,b['q0.5'],c['q0.95']], 1)



a['q0.05'] = a['q0.05'].clip(0, 10000)

a['q0.5'] = a['q0.5'].clip(0, 10000)

a['q0.95'] = a['q0.95'].clip(0, 10000)



a
a['Id'] = a['Id'] + 1

a
sub = pd.melt(a, id_vars=['Id'], value_vars = ['q0.05', 'q0.5', 'q0.95'])

sub['variable'] = sub['variable'].str.replace('q', '', regex=False)

sub['ForecastId_Quantile'] = sub['Id'].astype(str)+'-'+sub['variable']

sub['TargetValue'] = sub['value']

sub = sub[['ForecastId_Quantile', 'TargetValue']]

sub.reset_index(drop=True, inplace=True)

sub.to_csv('submission.csv', index=False)

sub.head()