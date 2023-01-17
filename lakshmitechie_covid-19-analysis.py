import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



%matplotlib inline
train = pd.read_csv('../input/covid19-global-forecasting-week-1/train.csv')

test = pd.read_csv('../input/covid19-global-forecasting-week-1/test.csv')
train.head()
test.tail()
train.dtypes
train['Date']= pd.to_datetime(train['Date']) 

test['Date']= pd.to_datetime(test['Date']) 
new_train = train.set_index(['Date'])

new_test = test.set_index(['Date'])
new_test.head()
new_train.head()
new_train.isnull().sum()
new_train.info()
new_test = new_test.drop(["ForecastId"], axis=1)
new_train = new_train.drop(["Id"], axis=1)
new_train[['Province/State']] = new_train[['Province/State']].fillna('')
new_train.isnull().sum()
new_test[['Province/State']] = new_test[['Province/State']].fillna('')
new_test.isnull().sum()
import plotly.express as px
# Creating a dataframe with total no of cases for every country

confirmiedcases = pd.DataFrame(new_train.groupby('Country/Region')['ConfirmedCases'].sum())

confirmiedcases['Country/Region'] = confirmiedcases.index
confirmiedcases.index
confirmiedcases.index = np.arange(1,164)
global_confirmiedcases = confirmiedcases[['Country/Region','ConfirmedCases']]
fig = px.bar(global_confirmiedcases.sort_values('ConfirmedCases',ascending=False)[:20][::-1],

             x='ConfirmedCases',y='Country/Region',title='Confirmed Cases Worldwide',text='ConfirmedCases', height=900, orientation='h')

fig.show()
formated_gdf = new_train.groupby(['Date', 'Country/Region'])['ConfirmedCases'].max()

formated_gdf = formated_gdf.reset_index()

formated_gdf['Date'] = pd.to_datetime(formated_gdf['Date'])

formated_gdf['Date'] = formated_gdf['Date'].dt.strftime('%m/%d/%Y')

formated_gdf['size'] = formated_gdf['ConfirmedCases'].pow(0.3)



fig = px.scatter_geo(formated_gdf, locations="Country/Region", locationmode='country names', 

                     color="ConfirmedCases", size='size', hover_name="Country/Region", 

                     range_color= [0, 1500], 

                     projection="natural earth", animation_frame="Date", 

                     title='CORONA: Spread Over Time From Jan 2020 to Mar 2020', color_continuous_scale="portland")

fig.show()
new_train.head()
new_test.head()
new_train["Country/Region"].unique()
def create_time_features(df):

    """

    Creates time series features from datetime index

    """

    df['date'] = df.index

    df['hour'] = df['date'].dt.hour

    df['dayofweek'] = df['date'].dt.dayofweek

    df['quarter'] = df['date'].dt.quarter

    df['month'] = df['date'].dt.month

    df['year'] = df['date'].dt.year

    df['dayofyear'] = df['date'].dt.dayofyear

    df['dayofmonth'] = df['date'].dt.day

    df['weekofyear'] = df['date'].dt.weekofyear

    

    X = df[['hour','dayofweek','quarter','month','year',

           'dayofyear','dayofmonth','weekofyear']]

    return X
create_time_features(new_train)

create_time_features(new_test)
new_train.head()
new_train.drop("date", axis=1, inplace=True)

new_test.drop("date", axis=1, inplace=True)
new_test.head()
new_train = pd.concat([new_train,pd.get_dummies(new_train['Province/State'], prefix='ps')],axis=1)

new_train.drop(['Province/State'],axis=1, inplace=True)

new_test = pd.concat([new_test,pd.get_dummies(new_test['Province/State'], prefix='ps')],axis=1)

new_test.drop(['Province/State'],axis=1, inplace=True)
new_train = pd.concat([new_train,pd.get_dummies(new_train['Country/Region'], prefix='cr')],axis=1)

new_train.drop(['Country/Region'],axis=1, inplace=True)

new_test = pd.concat([new_test,pd.get_dummies(new_test['Country/Region'], prefix='cr')],axis=1)

new_test.drop(['Country/Region'],axis=1, inplace=True)
new_train.head()
y_train = new_train["Fatalities"]
X_train = new_train.drop(["Fatalities", "ConfirmedCases"], axis=1)
import xgboost as xgb

from xgboost import plot_importance, plot_tree

from sklearn.metrics import mean_squared_error, mean_absolute_error
reg = xgb.XGBRegressor(n_estimators=1000)
reg.fit(X_train, y_train, verbose=True)
plot = plot_importance(reg, height=0.9, max_num_features=20)
y_train = train["ConfirmedCases"]
reg = xgb.XGBRegressor(n_estimators=1000)
reg.fit(X_train, y_train, verbose=True)
plot = plot_importance(reg, height=0.9, max_num_features=20)
y_train = train.groupby(["Country/Region"]).ConfirmedCases.pct_change(periods=1)
y_train = y_train.replace(np.nan, 0)
y_train = y_train.replace(np.inf, 0)
reg = xgb.XGBRegressor(n_estimators=1000)
reg.fit(X_train, y_train, verbose=True)
plot = plot_importance(reg, height=0.9, max_num_features=20)
y_train = train["ConfirmedCases"]

confirmed_reg = xgb.XGBRegressor(n_estimators=1000)

confirmed_reg.fit(X_train, y_train, verbose=True)

preds = confirmed_reg.predict(new_test)

preds = np.array(preds)

preds[preds < 0] = 0

preds = np.round(preds, 0)
preds = np.array(preds)
preds
submission_new = pd.read_csv("../input/covid19-global-forecasting-week-1/submission.csv")
submission_new.head()
submission_new["ConfirmedCases"]=pd.Series(preds)
y_train = train["Fatalities"]

confirmed_reg = xgb.XGBRegressor(n_estimators=1000)

confirmed_reg.fit(X_train, y_train, verbose=True)

preds = confirmed_reg.predict(new_test)

preds = np.array(preds)

preds[preds < 0] = 0

preds = np.round(preds, 0)

submission_new["Fatalities"]=pd.Series(preds)
submission_new
submission_new.to_csv('submission.csv',index=False)