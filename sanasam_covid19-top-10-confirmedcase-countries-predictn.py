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
## Importing Python libraries



import matplotlib.pyplot as plt

import seaborn as sns

sns.set()

import plotly.graph_objects as go

import plotly.express as px

import plotly.io as pio

from plotly.subplots import make_subplots

from datetime import datetime
train = pd.read_csv('../input/covid19-global-forecasting-week-1/train.csv')

train.head()
train.info()
train_confm_date=train.groupby('Date')['ConfirmedCases','Fatalities'].sum()
train_confm_date.head()
plt.figure(figsize=(16,10))

train_confm_date.plot()

plt.title('Globally COnfirmed case and Fatalities')

plt.xticks(rotation=60)
train_confirm_country=train.groupby('Country/Region')['ConfirmedCases','Fatalities'].sum().reset_index().sort_values('ConfirmedCases',ascending=False)
train_confirm_country.head()
plt.figure(figsize=(12,6))

plt.bar(train_confirm_country['Country/Region'][:10],train_confirm_country['ConfirmedCases'][:10])

plt.bar(train_confirm_country['Country/Region'][:10],train_confirm_country['Fatalities'][:10])

plt.legend([' Blue Color: Confirmed cases and Yellow Color : Fatality'])
train_confirm_country['Fatality rate in %']=train_confirm_country['Fatalities']/train_confirm_country['ConfirmedCases']
train_confirm_country.sort_values('Fatality rate in %', ascending=False).head(10)
train_confirm_country.head(10)
plt.figure(figsize=(12,6))

plt.bar(train_confirm_country['Country/Region'][:10],train_confirm_country['Fatality rate in %'][:10])
df_top_10=train_confirm_country[:10]

df_top_10.head(5)

sns.barplot(y='Country/Region',x='Fatality rate in %',data=df_top_10)
train_daily_report=train.groupby('Date').sum()



train_daily_report.head()
plt.figure(figsize=(18,10))

train_daily_report[['ConfirmedCases','Fatalities']].plot()

plt.xticks(rotation=60)
train_daily_report_china=train[train['Country/Region']=='China']

train_daily_report_china_sort=train_daily_report_china.groupby('Date')['ConfirmedCases','Fatalities'].sum()

plt.figure(figsize=(18,8))

train_daily_report_china_sort.plot()

plt.ylabel('No.of confirmed cases')

plt.legend(['China: COnfirmed cases till 2020-03-22'])

plt.xticks(rotation=60)
train_daily_report_india=train[train['Country/Region']=='India']

train_daily_report_india_sort=train_daily_report_india.groupby('Date')['ConfirmedCases','Fatalities'].sum()

plt.figure(figsize=(12,6))

train_daily_report_india_sort.plot()

plt.ylabel('No.of confirmed cases')

plt.legend(['India: COnfirmed cases till 2020-03-22'])

plt.xticks(rotation=60)
train_daily_report_italy=train[train['Country/Region']=='Italy']

train_daily_report_italy_sort=train_daily_report_italy.groupby('Date')['ConfirmedCases','Fatalities'].sum()

plt.figure(figsize=(12,6))

train_daily_report_italy_sort.plot()

plt.ylabel('No.of confirmed cases')

plt.legend(['Italy: COnfirmed cases till 2020-03-22'])

plt.xticks(rotation=60)

train_daily_report_iran=train[train['Country/Region']=='Iran']

train_daily_report_iran_sort=train_daily_report_iran.groupby('Date')['ConfirmedCases','Fatalities'].sum()

plt.figure(figsize=(12,6))

train_daily_report_iran_sort.plot()

plt.ylabel('No.of confirmed cases')

plt.legend(['Iran: COnfirmed cases till 2020-03-22'])

plt.xticks(rotation=60)
train.head()
test = pd.read_csv('../input/covid19-global-forecasting-week-1/test.csv')

test.head()
sns.heatmap(train.isnull(),yticklabels=False, cbar=False)
sns.heatmap(test.isnull(),yticklabels=False, cbar=False)
## Since we have Lat and long, we can drop province and country from the the dataset

train.drop(['Province/State','Country/Region'],axis=1,inplace=True)
test.drop(['Province/State','Country/Region'],axis=1,inplace=True)

display(train.info())

display(test.info())
train["Date"] = train["Date"].apply(lambda x: x.replace("-",""))

train['Date']=train['Date'].astype(int)

train.info()
test["Date"] = test["Date"].apply(lambda x: x.replace("-",""))

test["Date"]  = test["Date"].astype(int)

test.info()
X_train=train.drop(['Id','ConfirmedCases','Fatalities'],axis=1)

y_confrm=train[['ConfirmedCases']]

y_fat=train[['Fatalities']]

X_test=test.drop('ForecastId',axis=1)

X_test.head()
from sklearn.ensemble import RandomForestRegressor

rand_reg = RandomForestRegressor(random_state=42)

rand_reg.fit(X_train,y_confrm)



pred_grid1 = rand_reg.predict(X_test)

pred_grid1 = pd.DataFrame(pred_grid1).round()

pred_grid1.columns = ["ConfirmedCases_prediction"]

pred_grid1.head()
rand_reg.fit(X_train,y_fat)



pred_grid2 = rand_reg.predict(X_test)

pred_grid2 = pd.DataFrame(pred_grid2).round()

pred_grid2.columns = ["Fatality_prediction"]

pred_grid2.head()
sample=pd.read_csv('../input/covid19-global-forecasting-week-1/submission.csv')
submission=sample[['ForecastId']]

submission.head()
from sklearn.tree import DecisionTreeRegressor

tree_reg=DecisionTreeRegressor(random_state=42)

tree_reg.fit(X_train,y_confrm)
y_tree_conf=tree_reg.predict(X_test)

y_tree_conf=pd.DataFrame(y_tree_conf)

y_tree_conf.columns=['Confrmed_prediction']

y_tree_conf.head()
from sklearn.tree import DecisionTreeRegressor

tree_reg=DecisionTreeRegressor(random_state=42)

tree_reg.fit(X_train,y_fat)



y_tree_fat=tree_reg.predict(X_test)

y_tree_fat=pd.DataFrame(y_tree_fat).round()

y_tree_fat.columns=['fatality_prediction']

y_tree_fat.head()
final_sub_tree=pd.concat([submission,y_tree_conf,y_tree_fat],axis=1)

final_sub_tree.head()
final_sub_tree.columns=[['ForecastId','ConfirmedCases', 'Fatalities']]

final_sub_tree.head()
final_sub_tree.to_csv("submission.csv",index=False)