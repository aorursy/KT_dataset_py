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
import csv

import os

import xgboost



import re

import string

from sklearn import ensemble

from sklearn import metrics



import matplotlib.pyplot as plt

import seaborn as sns



import plotly.express as px

import plotly.graph_objs as go

import plotly.graph_objects as go

from plotly.subplots import make_subplots

import plotly.offline as pyo

pyo.init_notebook_mode()





from sklearn import preprocessing

from sklearn.metrics import confusion_matrix, roc_auc_score

from sklearn.model_selection import StratifiedKFold, cross_val_score, KFold

from xgboost import XGBClassifier

import xgboost as xgb





train = pd.read_csv("../input/covid19-global-forecasting-week-2/train.csv")

test = pd.read_csv("../input/covid19-global-forecasting-week-2/test.csv")

df_1 = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/2014_world_gdp_with_codes.csv')
print(train['Date'].min())

print(train['Date'].max())



print(test['Date'].min())

print(test['Date'].max())
train['Date'] = pd.to_datetime(train['Date'])

test['Date'] = pd.to_datetime(test['Date'])



train['dayofmonth'] = train['Date'].dt.day

train['dayofweek'] = train['Date'].dt.dayofweek

train['month'] = train['Date'].dt.month

train['weekNumber'] = train['Date'].dt.week

train['dayofyear'] = train['Date'].dt.dayofyear

## added in training set

train['Fatalities_ratio'] = train['Fatalities'] / train['ConfirmedCases']



#train['Change_ConfirmedCases'] = train.groupby('Country_Region').ConfirmedCases.pct_change()

#train['Change_Fatalities'] = train.groupby('Country_Region').Fatalities.pct_change()



## to deal with data wih Province State

train['Change_ConfirmedCases'] = train.groupby(np.where(train['Province_State'].isnull(), train['Country_Region'], train['Province_State'])).ConfirmedCases.pct_change()

train['Change_Fatalities'] = train.groupby(np.where(train['Province_State'].isnull(), train['Country_Region'], train['Province_State'])).Fatalities.pct_change()



## added in Test set

test['dayofmonth'] = test['Date'].dt.day

test['dayofweek'] = test['Date'].dt.dayofweek

test['month'] = test['Date'].dt.month

test['weekNumber'] = test['Date'].dt.week

test['dayofyear'] = test['Date'].dt.dayofyear
country_code_list = df_1[['COUNTRY', 'CODE']]

train_join_left = pd.merge(train, country_code_list, left_on='Country_Region', right_on='COUNTRY', how='left')

total_df1 = train_join_left.groupby(["COUNTRY","CODE", "Date"])["ConfirmedCases", "Fatalities"].sum().reset_index()



total_df1["Date"] = pd.to_datetime(total_df1["Date"] , format="%m/%d/%Y").dt.date

total_df1 = total_df1.sort_values(by="Date").reset_index(drop=True)

import datetime

start_date = datetime.date(2020,1, 22)

total_df1["Date"] = total_df1["Date"].astype(str)



fig = px.choropleth(total_df1, locations="CODE",

                    color="ConfirmedCases", 

                    animation_frame="Date",

                    color_continuous_scale='Blues',

                    range_color=[0,75000]

                    #autocolorscale=False,

                   )





fig.update_geos(

    resolution=50,

   # showcoastlines=True, coastlinecolor="Red",

    showland=True, landcolor="white",

    showocean=True, oceancolor="LightBlue",

    showlakes=True, lakecolor="LightBlue",

    showrivers=True, rivercolor="LightBlue"

)







layout = go.Layout(

    title=go.layout.Title(

        text="Total Confirmed COVID-19 Cases Around The World",

        x=0.5

    ),

    font=dict(size=12),

)





fig.update_layout(layout

                 ,height = 500)

fig.show()
fig = px.choropleth(total_df1, locations="CODE",

                    color="Fatalities", 

                    animation_frame="Date",

                    color_continuous_scale='Greens',

                    range_color=[0,7000]

                    #autocolorscale=False,

                   )



fig.update_geos(

    resolution=50,

 #   showcoastlines=True, coastlinecolor="Red",

    showland=True, landcolor="white",

    showocean=True, oceancolor="LightBlue",

    showlakes=True, lakecolor="LightBlue",

    showrivers=True, rivercolor="LightBlue"

)



layout = go.Layout(

    title=go.layout.Title(

        text="Total Fatalities from COVID-19 Around The World",

        x=0.5

    ),

    font=dict(size=12),

)



fig.update_layout(layout

                  , height = 500

                  #, width = 1000

                 )

fig.show()
train.groupby(['Country_Region', 'Province_State'])['ConfirmedCases', 'Fatalities'].mean().reset_index().tail(10)
cnty_state = train.groupby(['Country_Region','Province_State', 'Date'])['ConfirmedCases', 'Fatalities'].sum().reset_index()

cnty_state['ratio'] = (cnty_state['Fatalities'] / cnty_state['ConfirmedCases']) * 100

cnty_state.sort_values("ratio", axis = 0, ascending = True, inplace = True, na_position ='first')

cnty_state.sort_values("ratio", ascending = False).head(10)
total_by_date = train.groupby(['Date'])['ConfirmedCases', 'Fatalities'].sum().reset_index().sort_values('Date', ascending = True)

total_by_date['Fatality_ratio'] = total_by_date['Fatalities'] / total_by_date['ConfirmedCases']

total_by_date['CC_change_ratio'] = total_by_date['ConfirmedCases'].pct_change()

total_by_date['F_change_ratio'] = total_by_date['Fatalities'].pct_change()



fig = px.line(total_by_date, x="Date", y="ConfirmedCases", title='Total Cases of ')

fig = px.line(total_by_date, x="Date", y="Fatalities", title='Total Cases of ')

fig.add_scatter(x=total_by_date['Date'], y=total_by_date['ConfirmedCases'], mode='lines', name="Confirmed Cases", showlegend=True)

fig.add_scatter(x=total_by_date['Date'], y=total_by_date['Fatalities'], mode='lines', name="Fatality", showlegend=True)

fig.show()

fig = px.scatter(total_by_date, x="Date", y="ConfirmedCases")

fig = px.scatter(total_by_date, x="Date", y="Fatalities")

fig.add_scatter(x=total_by_date['Date'], y=total_by_date['ConfirmedCases'], mode='lines', name="Confirmed Cases", showlegend=True)

fig.add_scatter(x=total_by_date['Date'], y=total_by_date['Fatalities'], mode='lines', name="Total Fatalities", showlegend=True)

fig.show()
plt.figure(figsize=(25,15))



exp1= total_by_date['ConfirmedCases'].ewm(span=7, adjust=False).mean()

exp2= total_by_date['ConfirmedCases'].ewm(span=14, adjust=False).mean()



plt.plot(total_by_date.Date, total_by_date.ConfirmedCases, label='Confirmed Cases')

plt.plot(total_by_date.Date, exp1, label='Confirmed Cases 7 Days ')

plt.plot(total_by_date.Date, exp2, label='Confirmed Cases 14 Days ')

plt.legend(loc='upper left')

plt.title('Total Confirmed Cases with Exponential Moving Average by 1 week & 2 weeks')

plt.show()
plt.figure(figsize=(25,15))



exp1_f = total_by_date['Fatalities'].ewm(span=7, adjust=False).mean()

exp2_f = total_by_date['Fatalities'].ewm(span=14, adjust=False).mean()



plt.plot(total_by_date.Date, total_by_date.Fatalities, label='Total Fatalities')

plt.plot(total_by_date.Date, exp1_f, label='Confirmed Cases 7 Days ')

plt.plot(total_by_date.Date, exp2_f, label='Confirmed Cases 14 Days')

plt.legend(loc='upper left')

plt.title('Total Fatalities Cases with Exponential Moving Average by 1 week & 2 weeks')

plt.show()
fig = px.line(total_by_date, x="Date", y="CC_change_ratio", title='Total Cases of ')

fig = px.line(total_by_date, x="Date", y="F_change_ratio", title='Daily Change by Percentage (Confirmed Case and Fatalities) ')

fig.add_scatter(x=total_by_date['Date'], y=total_by_date['CC_change_ratio'], mode='lines', name="Confirmed Cases % change", showlegend=True)

fig.add_scatter(x=total_by_date['Date'], y=total_by_date['F_change_ratio'], mode='lines', name="Fatality % change", showlegend=True)

fig.show()
fig = go.Figure(data=go.Scatter(

    x=total_by_date['CC_change_ratio'],

    y=total_by_date['F_change_ratio'],

    mode='markers',

    marker=dict(size=[0,10,20,30,40,50,60,70,80, 90,100],

                color=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.9,1.0])

))



fig.update_layout(

    title="Bubble plot for change by % for fatalities and confirmed cases",

    xaxis_title="Confirmed Case % change",

    yaxis_title="Fatalities % change",

    font=dict(

  #      family="Courier New, monospace",

        size=15,

        color="#7f7f7f"

    )

)



fig.show()
fig = go.Figure(data=go.Scatter(

    y=total_by_date['CC_change_ratio'],

    x=total_by_date['F_change_ratio'],

    mode='markers',

    marker=dict(size=[0,10,20,30,40,50,60,70,80, 90,100],

                color=total_by_date['CC_change_ratio'])

))



fig.update_layout(

    title="Bubble plot for change by % for fatalities and confirmed cases",

    yaxis_title="Confirmed Case % change",

    xaxis_title="Fatalities % change",

    font=dict(

  #      family="Courier New, monospace",

        size=15,

        color="#7f7f7f"

    )

)

fig.show()
import plotly.express as px

#df = px.data.gapminder()



fig = px.scatter(train, x="ConfirmedCases", y="Fatalities",   

                 color="Country_Region",

                 hover_name="Province_State", log_x=True, size_max=60)

fig.show()
import plotly.express as px

df = px.data.gapminder()



fig = px.scatter(train, x="Date", y="ConfirmedCases",   

                 color="Country_Region",

                 hover_name="Province_State"

                 #, log_x=True, size_max=60

                )

                 

fig.show()
import plotly.express as px



fig = px.scatter(train, y="ConfirmedCases", x="Fatalities",   

                 color="Country_Region", 

                 log_x=True, size_max=60)

fig.show()
import plotly.express as px



fig = px.scatter(train.dropna(), y="ConfirmedCases", x="Fatalities",   

                 color="Province_State",

                 hover_name="Country_Region", 

                log_x=True, size_max=60

                )

fig.show()
fig = px.bar(train[train['ConfirmedCases'] > 0], x='Country_Region', y='ConfirmedCases',)

fig.show()
plt.figure(figsize=(25,15))

x = sns.barplot(x="Country_Region", y="ConfirmedCases", data=train[train['ConfirmedCases'] > 0], ci=None)

x.set_xticklabels(x.get_xticklabels(), rotation=90, horizontalalignment='center')

plt.title("Number of Confirmed Cases by Country between 2020-01-22 - 2020-03-26", size= 20)

plt.show()
plt.figure(figsize=(25,10))

x = sns.barplot(x="Country_Region", y="Fatalities", data=train[train['Fatalities'] > 0], ci=None)

x.set_xticklabels(x.get_xticklabels(), rotation=90, horizontalalignment='center')

plt.title("Number of Fatalities by Country between 2020-01-22 - 2020-03-26", size= 20)

plt.show()
train['dayofweek'] = train['Date'].dt.day_name()



plt.figure(figsize=(15,10))



x = sns.barplot(x="dayofweek", y="Fatalities", data=train[train['Country_Region'] == 'Italy'], ci=None)

x.set_xticklabels(x.get_xticklabels(), rotation=45, horizontalalignment='center')

plt.title("Number of Fatalities by Country between 2020-01-22 - 2020-03-26 in Italy", size= 20)

plt.show()
plt.figure(figsize=(20,10))



x = sns.barplot(x="dayofweek", y="Fatalities", data=train[train['Country_Region'] == 'US'], ci=None)

x.set_xticklabels(x.get_xticklabels(), rotation=45, horizontalalignment='center')

plt.title("Number of Fatalities by Country between 2020-01-22 - 2020-03-26 in USA", size= 20)



plt.show()
plt.figure(figsize=(15,10))



x = sns.barplot(x="dayofweek", y="Fatalities", data=train[train['Country_Region'] == 'Iran'], ci=None)

x.set_xticklabels(x.get_xticklabels(), rotation=45, horizontalalignment='center')

plt.title("Number of Fatalities by Country between 2020-01-22 - 2020-03-26 in IRAN", size= 20)

#x.get_xticklabels(rotation=90)

#plt.xticks(rotation=90)

plt.show()
plt.figure(figsize=(15,10))



x = sns.barplot(x="dayofweek", y="Fatalities", data=train[train['Country_Region'] == 'Spain'], ci=None)

x.set_xticklabels(x.get_xticklabels(), rotation=45, horizontalalignment='center')

plt.title("Number of Fatalities by Country between 2020-01-22 - 2020-03-26 in SPAIN", size= 20)

plt.show()
train['dayofweek'] = train['Date'].dt.dayofweek
from sklearn import preprocessing

le = preprocessing.LabelEncoder()

from xgboost import XGBRegressor





train['Country_Region'] = le.fit_transform(train['Country_Region'])

train['Province_State'] = le.fit_transform(train['Province_State'].fillna('0'))



test['Country_Region'] = le.fit_transform(test['Country_Region'])

test['Province_State'] = le.fit_transform(test['Province_State'].fillna('0'))



y1_train = train['ConfirmedCases']

y2_train = train['Fatalities']

X_Id = train['Id']



X_train = train.drop(columns=['Id', 'Date','ConfirmedCases', 'Fatalities', 'Fatalities_ratio','Change_ConfirmedCases','Change_Fatalities'])

X_test  = test.drop(columns=['ForecastId', 'Date'])





model = xgboost.XGBRegressor(colsample_bytree=0.4,

                 gamma=0,                 

                 learning_rate=0.07,

                 max_depth=3,

                 min_child_weight=1.5,

                 n_estimators=10000,                                                                    

                 reg_alpha=0.75,

                 reg_lambda=0.45,

                 subsample=0.6,

                 seed=42) 





model.fit(X_train, y1_train)

y1_pred = model.predict(X_test)





model.fit(X_train, y2_train)

y2_pred = model.predict(X_test)





df = pd.DataFrame({'ForecastId': test.ForecastId, 'ConfirmedCases': y1_pred, 'Fatalities': y2_pred})



# df.to_csv('submission.csv', index=False)
model_xgb_default = XGBRegressor(n_estimators=1000)

model_xgb_default.fit(X_train, y1_train)

y1_pred_xgb_d = model_xgb_default.predict(X_test)

model_xgb_default.fit(X_train, y2_train)

y2_pred_xgb_d = model_xgb_default.predict(X_test)

df_xgb_d = pd.DataFrame({'ForecastId': test.ForecastId, 'ConfirmedCases': y1_pred_xgb_d, 'Fatalities': y2_pred_xgb_d})



##

df_xgb_d.to_csv('submission.csv', index=False)
