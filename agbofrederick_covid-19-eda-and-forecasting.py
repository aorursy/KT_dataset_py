# Importing required libraries for data processing and visualization

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
pio.templates.default = "plotly_dark"
%matplotlib inline

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
df_train = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-3/train.csv', na_filter=False)
df_train.columns.tolist()
df_train=df_train.drop(['Id'],axis=1)

df_train.head(10)
df_train['Country_Region'].unique().tolist()
# Get the number of countries affected with the virus

affected_country = df_train['Country_Region'].nunique()
earliest_entry = f"{df_train['Date'].min()}"
last_entry = f"{df_train['Date'].max()}"
print('There are {a} number of affected countries within {b} and {c}'.format(a=affected_country, b=earliest_entry, c=last_entry))

# confirmed cases as of 03-04-2020

xy = df_train.drop('Province_State',axis=1)
current = xy[xy['Date'] == max(xy['Date'])].reset_index()
current_case = current.groupby('Country_Region')['ConfirmedCases','Fatalities'].sum().reset_index()
highest_case = current.groupby('Country_Region')['ConfirmedCases'].sum().reset_index()
fig = px.bar(highest_case.sort_values('ConfirmedCases', ascending=False)[:5][::-1], 
             x='ConfirmedCases', y='Country_Region',
             title='Global Confirmed Cases (03-04-2020)', text='ConfirmedCases', height=900, orientation='h')
fig.show()
# plot the confirmed cases in the world over time.

world_wide_case = df_train.groupby('Date')['ConfirmedCases'].sum().reset_index()
fig = px.line(world_wide_case, x="Date", y="ConfirmedCases", 
              title="Worldwide Confirmed Cases Over Time")
fig.show()
# Countries with the highest death rate

highest_death = current.groupby('Country_Region')['Fatalities'].sum().reset_index()
fig = px.bar(highest_death.sort_values('Fatalities',ascending=False)[:5][::-1],
            x='Fatalities',y='Country_Region',
             title='Global Death Cases (03-04-2020)', text='Fatalities', height=900, orientation='h')
fig.show()
# Death cases worldwide over time

death_cases = df_train.groupby('Date')['Fatalities'].sum().reset_index()
fig = px.line(death_cases, x="Date", y="Fatalities", 
              title="Worldwide Fatalities Over Time")
fig.show()
# How did covid-19 spread?

virus_spread = df_train.groupby(['Date', 'Country_Region'])['ConfirmedCases', 'Fatalities'].max()
virus_spread = virus_spread.reset_index()
virus_spread['Date'] = pd.to_datetime(virus_spread['Date'])
virus_spread['Date'] = virus_spread['Date'].dt.strftime('%m/%d/%Y')
virus_spread['Size'] = virus_spread['ConfirmedCases'].pow(0.3)
fig = px.scatter_geo(virus_spread, locations="Country_Region", locationmode='country names', color="ConfirmedCases", size='Size', hover_name="Country_Region", range_color= [0, 100], projection="natural earth", animation_frame="Date", title='COVID-19: Virus Spread Over Time Globally (2020–01–22 to 2020–03–30.)', color_continuous_scale="peach")
fig.show()
df_test = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-3/test.csv")
df_test.head()
test_data = (
    df_test.groupby(["Date", "Country_Region"]).last().reset_index()[["Date", "Country_Region"]])
test_data
# importing required libraries for data processing and prediction

from sklearn.preprocessing import OrdinalEncoder
from sklearn import metrics
import xgboost as xgb
from xgboost import XGBRegressor
from xgboost import plot_importance, plot_tree
import datetime as dt
def categoricalToInteger(df):
    #convert NaN Province State values to a string
    df.Province_State.fillna('NaN', inplace=True)
    #Define Ordinal Encoder Model
    oe = OrdinalEncoder()
    df[['Province_State','Country_Region']] = oe.fit_transform(df.iloc[:,1:3])
    return df
def create_features(df):
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df['day'] = df['Date'].dt.day
    df['month'] = df['Date'].dt.month
    df['dayofweek'] = df['Date'].dt.dayofweek
    df['dayofyear'] = df['Date'].dt.dayofyear
    df['quarter'] = df['Date'].dt.quarter
    df['weekofyear'] = df['Date'].dt.weekofyear
    return df
def cum_sum(df, date, country, state):
    sub_df = df[(df['Country_Region']==country) & (df['Province_State']==state) & (df['Date']<=date)]
    display(sub_df)
    return sub_df['ConfirmedCases'].sum(), sub_df['Fatalities'].sum()
# Split the training data into train and dev set for cross-validation.

def train_dev_split(df):
    date = df['Date'].max() - dt.timedelta(days=7)
    return df[df['Date'] <= date], df[df['Date'] > date]
df_train = categoricalToInteger(df_train)
df_train = create_features(df_train)
df_train, df_dev = train_dev_split(df_train)
# Selecting all columns that are necessary for prediction

columns = ['day','month','dayofweek','dayofyear','quarter','weekofyear','Province_State', 'Country_Region','ConfirmedCases','Fatalities']
df_train = df_train[columns]
df_dev = df_dev[columns]
# Training and evaluating modeling

train = df_train.values
dev = df_dev.values
X_train, y_train = train[:,:-2], train[:,-2:]
X_dev, y_dev = dev[:,:-2], dev[:,-2:]
'''train = df_train.values
X_train, y_train = train[:,:-2], train[:,-2:]'''
def modelfit(alg, X_train, y_train,useTrainCV=True, cv_folds=5, early_stopping_rounds=50):
    
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(X_train, label=y_train)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
            metrics='rmse', early_stopping_rounds=early_stopping_rounds, show_stdv=False)
        alg.set_params(n_estimators=cvresult.shape[0])
    
    #Fit the algorithm on the data
    alg.fit(X_train, y_train,eval_metric='rmse')
        
    #Predict training set:
    predictions = alg.predict(X_train)
    #predprob = alg.predict_proba(X_train)[:,1]
        
    #Print model report:
    print("\nModel Report")
    #print("Accuracy : %.4g" % metrics.accuracy_score(y_train, predictions))
    print("RMSE Score (Train): %f" % metrics.mean_squared_error(y_train, predictions))
                    
    feat_imp = pd.Series(alg.feature_importances_).sort_values(ascending=False)
    feat_imp.plot(kind='bar', title='Feature Importances')
    plt.ylabel('Feature Importance Score')
'''model1 = XGBRegressor(
 learning_rate =0.1,
 n_estimators=1000,
 max_depth=5,
 min_child_weight=1,
 gamma=0,
 subsample=0.8,
 colsample_bytree=0.8,
 objective= 'reg:squarederror',
 scale_pos_weight=1)
modelfit(model1, X_train, y_train[:,0])'''
'''model2 = XGBRegressor(
 learning_rate =0.1,
 n_estimators=1000,
 max_depth=5,
 min_child_weight=1,
 gamma=0,
 subsample=0.8,
 colsample_bytree=0.8,
 objective= 'reg:squarederror',
 scale_pos_weight=1)
modelfit(model2, X_train, y_train[:,1])'''
# Creating the model

model1 = XGBRegressor(n_estimators=1000)
model2 = XGBRegressor(n_estimators=1000)
# training the model

model1.fit(X_train, y_train[:,0],
           eval_set=[(X_train, y_train[:,0]), (X_dev, y_dev[:,0])],
           verbose=False)
model2.fit(X_train, y_train[:,1],
           eval_set=[(X_train, y_train[:,1]), (X_dev, y_dev[:,1])],
           verbose=False)
plot_importance(model1);
plot_importance(model2);
df_train = categoricalToInteger(df_test)
df_train = create_features(df_test)
columns = ['day','month','dayofweek','dayofyear','quarter','weekofyear','Province_State', 'Country_Region']
df_test = df_test[columns]
y_pred1 = model1.predict(df_test.values)
y_pred2 = model2.predict(df_test.values)
df_submit = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-3/submission.csv')
df_submit.ConfirmedCases = y_pred1
df_submit.Fatalities = y_pred2
'''df_submit.ConfirmedCases = df_submit.ConfirmedCases.apply(lambda x:max(0,round(x,0)))
df_submit.Fatalities = df_submit.Fatalities.apply(lambda x:max(0,round(x,0)))'''

df_submit.to_csv(r'submission.csv', index=False)