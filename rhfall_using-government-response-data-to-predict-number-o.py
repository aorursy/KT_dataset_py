import pandas as pd

import datetime

import math

import numpy as np
oxford = pd.read_csv("../input/government-response-to-covid19-worldwide/OxCGRT_Download_29_03.csv", sep=';')
oxford.head()
oxford['Date'] = pd.to_datetime(oxford['Date'], format='%Y%m%d')
oxford.drop(['CountryCode'], axis=1,inplace=True)
new_col = [x for x in oxford.columns if ('_Notes' not in x) and ('IsGeneral' not in x)] 
new_oxford = oxford[new_col]
new_oxford.drop(['S8_Fiscal measures', 'S9_Monetary measures', 'S10_Emergency investment in health care',

                           'S11_Investment in Vaccines', 'Unnamed: 34'], axis=1,inplace=True)
new_oxford[new_oxford['CountryName']=='Argentina']
new_oxford[new_oxford['CountryName']=='Norway']
countries = set(new_oxford['CountryName'].values)
oxford_fixed = pd.DataFrame(columns = new_oxford.columns)
for country in countries:

    df = new_oxford[new_oxford['CountryName'] == country].copy()

    #Cases

    for j in range(9,11):

        for i in range(df.shape[0]):

            if math.isnan(df.iloc[i, j]):

                df.iloc[i, j] = 0

            elif df.iloc[i, j] > 0:

                break

    #Measures

    for j in range(2,9):

        started = False

        for i in range(df.shape[0]):

            if started == False:

                if math.isnan(df.iloc[i, j]):

                    df.iloc[i, j] = 0

                elif df.iloc[i, j] > 0:

                    started = True

                    value = df.iloc[i, j]

            else:

                if math.isnan(df.iloc[i, j]):

                    df.iloc[i, j] = value

                else:

                    value = df.iloc[i, j]

    #Index

    j=11                

    started = False

    for i in range(df.shape[0]):

        if started == False:

            if math.isnan(df.iloc[i, j]):

                df.iloc[i, j] = 0

            elif df.iloc[i, j] > 0:

                started = True

                value = df.iloc[i, j]

        else:

            if math.isnan(df.iloc[i, j]):

                df.iloc[i, j] = value

            else:

                value = df.iloc[i, j]

    oxford_fixed = oxford_fixed.append(df)



oxford_fixed = pd.get_dummies(oxford_fixed, columns = new_oxford.columns[2:9],drop_first=True)
strigency = oxford_fixed[['StringencyIndex']] #saving for eventual use
oxford_fixed=oxford_fixed.drop(['StringencyIndex'],axis=1)
oxford_fixed.head()
list_days = ['Days_since_' + x for x in list(oxford_fixed.columns[2:])]
list_days
oxford_days = pd.DataFrame(columns=list_days, index=oxford_fixed.index).fillna(0)
list_perc = ['%ConfirmedCases', '%ConfirmedCases_smooth', '%ConfirmedDeaths', '%ConfirmedDeaths_smooth', 'Days', 'Days_since_100Cases']

oxford_perc = pd.DataFrame(columns=list_perc, index=oxford_fixed.index).fillna(0)
oxford_complete = pd.concat([oxford_fixed,oxford_days, oxford_perc],axis=1)
oxford_complete.head()
oxford_complete_days = pd.DataFrame(columns = oxford_complete.columns)

smooth = 4

for country in countries:

    df = oxford_complete[oxford_complete['CountryName'] == country].copy()

    #Measures

    for j in range(2,18):

        value = 0

        for i in range(df.shape[0]):

            df.iloc[i, j+16] = value

            if df.iloc[i, j] > 0:

                value = value + 1

    valueconf = 0

    valuedeath = 0

    days = 0

    days_100 = 0

    for i in range(df.shape[0]):

        

        if df.iloc[i, 2] > 0:

            valueconf = valueconf + 1

        if df.iloc[i, 3] > 0:

            valuedeath = valuedeath + 1

            

        if valueconf > 1:

            df.iloc[i,34] = (df.iloc[i,2] - df.iloc[i-1,2])/df.iloc[i-1,2]

        if valuedeath > 1:

            df.iloc[i,36] = (df.iloc[i,3] - df.iloc[i-1,3])/df.iloc[i-1,3]

            

        if valueconf >= smooth:

            df.iloc[i,35] = df.iloc[i-smooth+1:i+1,34].sum()/smooth

        if valuedeath >= smooth:

            df.iloc[i,37] = df.iloc[i-smooth+1:i+1,36].sum()/smooth 

        

        df.iloc[i,38] = days

        days = days + 1

        

        

        if df.iloc[i, 2] > 100:

            days_100 += days_100

        df.iloc[i,39] = days_100

        

        

    oxford_complete_days = oxford_complete_days.append(df)
oxford_complete_days.dropna(inplace=True) 
oxford_complete_days.head()
oxford_fixed.columns
rows_not_zero = [not x for x in oxford_complete_days[oxford_fixed.columns[2:]].sum(axis=1)==0]
oxford_complete_final = oxford_complete_days[rows_not_zero]
oxford_complete_final.head()
oxford_complete_final = oxford_complete_final.drop(oxford_complete_final.columns[4:18],axis=1)
oxford_complete_final.head()
strin = [not math.isnan(x) for x in new_oxford['StringencyIndex'].values]
new_oxford_drop_strin = new_oxford[strin]
new_oxford_drop_strin['CountryName'].value_counts().iloc[-40:]
new_oxford[new_oxford['CountryName'] == 'Iran'].iloc[-40:,:]
new_oxford[new_oxford['CountryName'] == 'Israel'].iloc[-40:,:]
list_countries = new_oxford_drop_strin['CountryName'].value_counts()>60
list_restricted = [x in list_countries.index[list_countries] for x in oxford_complete_final['CountryName'].values]
#list_test = ['Italy','United States','Spain','France','Germany']

#list_restricted = [x in list_test for x in oxford_complete_final['CountryName'].values]
oxford_restricted_final = oxford_complete_final[list_restricted]
oxford_restricted_final.shape
oxford_restricted_final.head()
#oxford_restricted_final.to_csv(r'~\COVID_gov_restricted_28_03.csv')
#oxford_complete_final.to_csv(r'~\COVID_gov_complete_28_03.csv')
date = datetime.datetime(2020,3,16,0,0) #Limit data that we will use to train the model
cols_y = ['ConfirmedCases', 'ConfirmedDeaths', '%ConfirmedCases', '%ConfirmedCases_smooth', '%ConfirmedDeaths', '%ConfirmedDeaths_smooth']
X = oxford_restricted_final.drop(cols_y, axis=1)

#X = oxford_restricted_final[['CountryName','Date','Days_since_ConfirmedCases', 'Days_since_ConfirmedDeaths','Days', 'Days_since_100Cases']]

y = oxford_restricted_final[cols_y+['Date']]

X_countries = X #Used later for ploting
X = pd.get_dummies(X, columns = ['CountryName'],drop_first=True)
X_train = X[X['Date']<date]

X_test = X[X['Date']>=date]



X_train_countries = X_countries[X_countries['Date']<date]

X_test_countries = X_countries[X_countries['Date']>=date]



X_train.drop(['Date'], axis=1,inplace=True)

X_test.drop(['Date'], axis=1,inplace=True)
y_train_death = y[y['Date']<date][['ConfirmedDeaths']]

y_test_death = y[y['Date']>=date][['ConfirmedDeaths']]



y_train_cases = y[y['Date']<date][['ConfirmedCases']]

y_test_cases = y[y['Date']>=date][['ConfirmedCases']]



y_train_death_rate = y[y['Date']<date][['%ConfirmedDeaths_smooth']]

y_test_death_rate = y[y['Date']>=date][['%ConfirmedDeaths_smooth']]



y_train_cases_rate = y[y['Date']<date][['%ConfirmedCases_smooth']]

y_test_cases_rate = y[y['Date']>=date][['%ConfirmedCases_smooth']]
#Imports de modelagem

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, auc, roc_curve, matthews_corrcoef

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, median_absolute_error



from sklearn.preprocessing import MinMaxScaler,StandardScaler

from sklearn.metrics import confusion_matrix

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, cross_val_score, RandomizedSearchCV

from math import sqrt



#Algorithms

from sklearn import ensemble, tree, svm, naive_bayes, neighbors, linear_model, gaussian_process, neural_network

import xgboost as xgb

from xgboost.sklearn import XGBClassifier
def metrics_report(X_test, y_test, reg):

    y_pred = reg.predict(X_test)

    return {'r2_score': r2_score(y_test, y_pred), 

          'mean_absolute_error': mean_absolute_error(y_test, y_pred),

          'mean_squared_error': mean_squared_error(y_test, y_pred),

         'median_absolute_error': median_absolute_error(y_test, y_pred),

           'RMSE': sqrt(mean_absolute_error(y_test, y_pred))}
from sklearn.ensemble import RandomForestRegressor
rfr_cases = RandomForestRegressor(n_estimators=300, n_jobs=-1, random_state=0)
rfr_cases.fit(X_train,y_train_cases)
metrics_report(X_test, y_test_cases, rfr_cases)
metrics_report(X_train, y_train_cases, rfr_cases)
pred = rfr_cases.predict(X_test)



pred_train = rfr_cases.predict(X_train)



y_pred_cases = pd.DataFrame(data = pred, index=y_test_cases.index, columns = y_test_cases.columns)



y_pred_cases_train = pd.DataFrame(data = pred_train, index=y_train_cases.index, columns = y_train_cases.columns)
import plotly.express as px

import plotly.graph_objects as go

from plotly.subplots import make_subplots
#countries = set(oxford_restricted_final['CountryName'])

countries = ['Italy','United States','Spain','France','Germany'] #Let's just plot some to make the visualization easier
fig = go.Figure()

for country in countries:

    df = X_test_countries[X_test_countries['CountryName']==country]

    fig.add_trace(go.Scatter(x=df['Days_since_ConfirmedCases'].values, y=y_test_cases.loc[df.index,'ConfirmedCases'].values,

            mode='lines+markers',

            name=country))

    fig.add_trace(go.Scatter(x=df['Days_since_ConfirmedCases'].values, y=y_pred_cases.loc[df.index,'ConfirmedCases'].values,

            mode='markers',

            marker_symbol='x',                     

            name=country + ' predicted'))

fig.update_layout(

    title="Predicted vs Real cases for test set",

    xaxis_title="Number of days since first confirmed case",

    yaxis_title="Number of cases",

)

fig.show()
fig = go.Figure()

for country in countries:

    df = X_train_countries[X_train_countries['CountryName']==country]

    fig.add_trace(go.Scatter(x=df['Days_since_ConfirmedCases'].values, y=y_train_cases.loc[df.index,'ConfirmedCases'].values,

            mode='lines+markers',

            name=country))

    fig.add_trace(go.Scatter(x=df['Days_since_ConfirmedCases'].values, y=y_pred_cases_train.loc[df.index,'ConfirmedCases'].values,

            mode='markers',

            marker_symbol='x',                     

            name=country + ' predicted'))

fig.update_layout(

    title="Predicted vs Real cases for training set",

    xaxis_title="Number of days since first confirmed case",

    yaxis_title="Number of cases",

)

fig.show()
!pip install treeinterpreter
from treeinterpreter import treeinterpreter as ti
prediction, bias, contributions = ti.predict(rfr_cases, X_test)
X_test_countries_italy = X_test_countries.reset_index()
X_test_countries_italy[X_test_countries_italy['CountryName']=='Italy']
italy_indexes = list(X_test_countries_italy[X_test_countries_italy['CountryName']=='Italy'].index)
for i in italy_indexes:

    print("Instance", i)

    print("Prediction: ", prediction[i])

    print("Bias (trainset mean)", bias[i])

    print("Feature contributions:")

    for c, feature in sorted(zip(contributions[i], 

                                 X_test.columns), 

                             key=lambda x: -abs(x[0]))[0:10]:

        print(feature, round(c, 2))

    print("-"*20)
X_train_Italy = X_train[X_train['CountryName_Italy']==1]
prediction_train, bias_train, contributions_train = ti.predict(rfr_cases, X_train_Italy)
for i in range(50,53):

    print("Instance", i)

    print("Prediction: ", prediction_train[i])

    print("Bias (trainset mean)", bias_train[i])

    print("Feature contributions:")

    for c, feature in sorted(zip(contributions_train[i], 

                                 X_train.columns), 

                             key=lambda x: -abs(x[0]))[:10]:

        print(feature, round(c, 2))

    print("-"*20)
#Feature importances

feature_importances = pd.DataFrame(rfr_cases.feature_importances_,

                                   index = X_train.columns,

                                    columns=['importance']).sort_values('importance',ascending=False)
feature_importances.iloc[:10]
rfr_cases_rate = RandomForestRegressor(n_estimators=300, n_jobs=-1, random_state=0)
rfr_cases_rate.fit(X_train,y_train_cases_rate)
metrics_report(X_test, y_test_cases_rate, rfr_cases_rate)
metrics_report(X_train, y_train_cases_rate, rfr_cases_rate)
pred_rate = rfr_cases_rate.predict(X_test)



pred_train_rate = rfr_cases_rate.predict(X_train)



y_pred_cases_rate = pd.DataFrame(data = pred_rate, index=y_test_cases_rate.index, columns = y_test_cases_rate.columns)



y_pred_cases_train_rate = pd.DataFrame(data = pred_train_rate, index=y_train_cases_rate.index, columns = y_train_cases_rate.columns)
fig = go.Figure()

for country in countries:

    df = X_test_countries[X_test_countries['CountryName']==country]

    fig.add_trace(go.Scatter(x=df['Days_since_ConfirmedCases'].values, y=y_test_cases_rate.loc[df.index,'%ConfirmedCases_smooth'].values,

            mode='lines+markers',

            name=country))

    fig.add_trace(go.Scatter(x=df['Days_since_ConfirmedCases'].values, y=y_pred_cases_rate.loc[df.index,'%ConfirmedCases_smooth'].values,

            mode='markers',

            marker_symbol='x',                     

            name=country + ' predicted'))

fig.update_layout(

    title="Predicted vs Real cases rate for test set",

    xaxis_title="Number of days since first confirmed case",

    yaxis_title="Rate of change of the number of cases",

)

fig.show()
fig = go.Figure()

for country in countries:

    df = X_train_countries[X_train_countries['CountryName']==country]

    fig.add_trace(go.Scatter(x=df['Days_since_ConfirmedCases'].values, y=y_train_cases_rate.loc[df.index,'%ConfirmedCases_smooth'].values,

            mode='lines+markers',

            name=country))

    fig.add_trace(go.Scatter(x=df['Days_since_ConfirmedCases'].values, y=y_pred_cases_train_rate.loc[df.index,'%ConfirmedCases_smooth'].values,

            mode='markers',

            marker_symbol='x',                     

            name=country + ' predicted'))

fig.update_layout(

    title="Predicted vs Real cases rate for train set",

    xaxis_title="Number of days since first confirmed case",

    yaxis_title="Rate of change of the number of cases",

)

fig.show()
prediction_rate, bias_rate, contributions_rate = ti.predict(rfr_cases_rate, X_test)
for i in italy_indexes:

    print("Instance", i)

    print("Prediction: ", prediction_rate[i])

    print("Bias (trainset mean)", bias_rate[i])

    print("Feature contributions:")

    for c, feature in sorted(zip(contributions_rate[i], 

                                 X_test.columns), 

                             key=lambda x: -abs(x[0]))[0:10]:

        print(feature, round(c, 2))

    print("-"*20)
prediction_train_rate, bias_train_rate, contributions_train_rate = ti.predict(rfr_cases_rate, X_train_Italy)
#Anlyzing form the beginning of the peak till the end of it

for i in range(28,45):

    print("Instance", i)

    print("Prediction: ", prediction_train_rate[i])

    print("Bias (trainset mean)", bias_train_rate[i])

    print("Feature contributions:")

    for c, feature in sorted(zip(contributions_train_rate[i], 

                                 X_train.columns), 

                             key=lambda x: -abs(x[0]))[:10]:

        print(feature, round(c, 2))

    print("-"*20)
#Feature importances

feature_importances = pd.DataFrame(rfr_cases_rate.feature_importances_,

                                   index = X_train.columns,

                                    columns=['importance']).sort_values('importance',ascending=False)
feature_importances.iloc[:10]
rfr_deaths = RandomForestRegressor(n_estimators=300, n_jobs=-1, random_state=0)
rfr_deaths.fit(X_train,y_train_death)
metrics_report(X_test, y_test_death, rfr_deaths)
metrics_report(X_train, y_train_death, rfr_deaths)
#Feature importances

feature_importances = pd.DataFrame(rfr_deaths.feature_importances_,

                                   index = X_train.columns,

                                    columns=['importance']).sort_values('importance',ascending=False)
feature_importances.iloc[:10]