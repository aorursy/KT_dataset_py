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





train = pd.read_csv("../input/covid19-global-forecasting-week-3/train.csv")

test = pd.read_csv("../input/covid19-global-forecasting-week-3/test.csv")

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
#train = train[train.Date<'2020-03-26']
enriched = pd.read_csv("/kaggle/input/data-prep-week3/enriched_covid_19_week_3.csv")

enriched['Date'] = pd.to_datetime(train['Date'])

enriched['Date'] = pd.to_datetime(test['Date'])

enriched["quarantine"] = pd.to_datetime(enriched["quarantine"])

enriched["publicplace"] = pd.to_datetime(enriched["publicplace"])

enriched["gathering"] = pd.to_datetime(enriched["gathering"])

enriched["nonessential"] = pd.to_datetime(enriched["nonessential"])

enriched["schools"] = pd.to_datetime(enriched["schools"])

enriched["firstcase"] = pd.to_datetime(enriched["firstcase"])

enriched["lockdown"] = pd.to_datetime(enriched["lockdown"])





dates_info = ["publicplace", "gathering", "nonessential", "quarantine", "schools","firstcase",'lockdown']



enriched["age_40-59"] = enriched.loc[:,["age_40-44", "age_45-49", "age_50-54", "age_55-59"]].values.sum(1)

enriched["age_60-79"] = enriched.loc[:,["age_60-64", "age_65-69", "age_70-74", "age_75-79"]].values.sum(1)

enriched["age_80+"]  = enriched.loc[:,["age_80-84", "age_85-89", "age_90-94", "age_95-99","age_100+"]].values.sum(1)



enriched.drop([

    "age_0-4", "age_5-9", "age_10-14", "age_15-19", "age_20-24", "age_25-29", "age_30-34", "age_35-39",

    "age_40-44", "age_45-49", "age_50-54", "age_55-59", "age_60-64", "age_65-69", "age_70-74", "age_75-79",

    "age_80-84", "age_85-89", "age_90-94","age_95-99","age_100+", "femalelung", "malelung"], 

    axis = 1, inplace = True)



enriched.head()



enriched = enriched.iloc[:,:]

enriched.info()
enriched[enriched.Country_Region=='Ukraine'].info()
def concat_country_province(country, province):

    if not isinstance(province, str):

        return country

    else:

        return country+"_"+province



# Concatenate region and province for training

train["Country_Region_"] = train[["Country_Region", "Province_State"]].apply(lambda x : concat_country_province(x[0], x[1]), axis=1)

test["Country_Region_"] = test[["Country_Region", "Province_State"]].apply(lambda x : concat_country_province(x[0], x[1]), axis=1)



enriched["Country_Region_"] = enriched[["Country_Region", "Province_State"]].apply(lambda x : concat_country_province(x[0], x[1]), axis=1)

enriched = enriched.drop_duplicates(subset=['Country_Region_'], keep="first", inplace=False)
enriched.iloc[:, 6:]
train = train.merge(enriched.iloc[:, 6:], on ='Country_Region_', how='left')

test = test.merge(enriched.iloc[:, 6:], on ='Country_Region_', how='left')
test.info()
train[train.Country_Region=='Ukraine']
def dates_diff_days(date_curr, date_):

    if date_curr>date_:

        return (date_curr - date_).days

    else :

        return 0





for col in dates_info:

    #print(merged.shape)

    train[col] =train[["Date", col]].apply(lambda x : dates_diff_days(x[0], x[1]), axis=1)  

    test[col] =test[["Date", col]].apply(lambda x : dates_diff_days(x[0], x[1]), axis=1) 



print(test.shape)



#drop_country_cols = [x for x in merged.columns if x.startswith("country")] + dates_info
test.shape
from sklearn import preprocessing

le = preprocessing.LabelEncoder()

from xgboost import XGBRegressor





train['ConfirmedCases_diff'] = train.loc[:, ['ConfirmedCases', 'Country_Region_']].groupby('Country_Region_').diff().fillna('0')

train['Fatalities_diff'] = train.loc[:, ['Fatalities', 'Country_Region_']].groupby('Country_Region_').diff().fillna('0')



train = train.astype({'ConfirmedCases_diff': 'int64','Fatalities_diff': 'int64' })



train['Country_Region'] = le.fit_transform(train['Country_Region'])

train['Province_State'] = le.fit_transform(train['Province_State'].fillna('0'))



test['Country_Region'] = le.fit_transform(test['Country_Region'])

test['Province_State'] = le.fit_transform(test['Province_State'].fillna('0'))



y1_train = train['ConfirmedCases_diff']

y2_train = train['Fatalities_diff']

X_Id = train['Id']



# X_train = train.drop(columns=['Id', 'Date','ConfirmedCases', 'Fatalities', 'Fatalities_ratio','Change_ConfirmedCases','Change_Fatalities'])

# X_test  = test.drop(columns=['ForecastId', 'Date'])



X_train = train.drop(columns=['Id', 'Fatalities', 'Date',

                              'Fatalities_ratio','Change_ConfirmedCases'

                              ,'Change_Fatalities','Country_Region_','ConfirmedCases','Fatalities_diff','ConfirmedCases_diff'])

X_test  = test.drop(columns=['ForecastId','Country_Region_', 'Date'])



X_train.info()
# from fbprophet import Prophet
# X_train = train.drop(columns=['Id', 'Fatalities', 'Fatalities_ratio','Change_ConfirmedCases','Change_Fatalities'])

# X_test  = test.drop(columns=['ForecastId'])



# model=Prophet()

# model.fit(X_train \

#               .rename(columns={'Date':'ds',

#                                'ConfirmedCases':'y'}))

# forecast_conf=model.predict(df=X_test \

#                                    .rename(columns={'Date':'ds'}))
# X_train = train.drop(columns=['Id', 'ConfirmedCases', 'Fatalities_ratio','Change_ConfirmedCases','Change_Fatalities'])

# X_test  = test.drop(columns=['ForecastId'])



# model_1=Prophet()

# model_1.fit(X_train \

#               .rename(columns={'Date':'ds',

#                                'Fatalities':'y'}))

# forecast_Fatilities=model.predict(df=X_test \

#                                    .rename(columns={'Date':'ds'}))
# df_xgb_d = pd.DataFrame({'ForecastId': test.ForecastId, 'ConfirmedCases': forecast_conf.yhat, 'Fatalities': forecast_Fatilities.yhat })

# df_xgb_d.to_csv('submission.csv', index=False)
X_train.info()
# X_train = train.drop(columns=['Id', 'Date','ConfirmedCases', 'Fatalities', 'Fatalities_ratio','Change_ConfirmedCases','Change_Fatalities'])

# X_test  = test.drop(columns=['ForecastId', 'Date'])



model = xgboost.XGBRegressor(colsample_bytree=0.7,

                 gamma=0,                 

                 learning_rate=0.1,

                 max_depth=6,

                 min_child_weight=1.5,

                 n_estimators=1000,                                                                    

                 reg_alpha=0.75,

                 reg_lambda=0.45,

                 subsample=0.5,

                 seed=42) 





model.fit(X_train, y1_train)

y1_pred = model.predict(X_test)





model.fit(X_train, y2_train)

y2_pred = model.predict(X_test)





df = pd.DataFrame({'ForecastId': test.ForecastId, 'ConfirmedCases': y1_pred, 'Fatalities': y2_pred})



import matplotlib.pylab as plt

from matplotlib import pyplot

from xgboost import plot_importance

plot_importance(model, max_num_features=20) # top 10 most important features

plt.show()
#df.to_csv('submission.csv', index=False)
test1 = test.copy()

test1['ConfirmedCases'] = y1_pred

test1['Fatalities']=y2_pred

train_max = train[['Country_Region_', 'ConfirmedCases','Fatalities']].groupby('Country_Region_').max().add_prefix('max_').reset_index()

train_max.head()
train_max = train.loc[train.Date == '2020-03-25',['Country_Region_', 'ConfirmedCases','Fatalities']]

train_max.rename(columns={"ConfirmedCases": "max_ConfirmedCases", 'Fatalities':'max_Fatalities',

                                 }, inplace=True)

#.add_prefix('max_')

train_max.head()
test1 = test1.merge(train_max, on='Country_Region_')
test1.loc[test1.ConfirmedCases<0, 'ConfirmedCases']=0

test1.loc[test1.Fatalities<0, 'Fatalities']=0
test1['ConfirmedCases'] = test1.groupby('Country_Region_')['ConfirmedCases'].cumsum()

test1['Fatalities'] = test1.groupby('Country_Region_')['Fatalities'].cumsum()

test1['ConfirmedCases'] = test1['ConfirmedCases'] + test1['max_ConfirmedCases']

test1['Fatalities'] = test1['Fatalities'] + test1['max_Fatalities']
test1.head()
#test1.loc[test1.Country_Region_=='Ukraine','femalelung':]
test1.loc[test1.Country_Region_=='Ukraine',[ 'Date', 'ConfirmedCases', 'Fatalities']]
subm_fit_log = pd.read_csv('../input/covid-19-fit-logistic-curves-and-submit-week-3/submission.csv')

subm_fit_log.rename(columns={"ConfirmedCases": "ConfirmedCases_log", 'Fatalities':'Fatalities_log',

                                 }, inplace=True)

subm_fit_log.head()

test1_ = test1.merge(subm_fit_log, on ='ForecastId')

test1_.head()
cols = ["Date"] + list(test1_.loc[:,'ConfirmedCases':].columns)
#test1.loc[test1.Country_Region_=='Italy',:]
test1_['ConfirmedCases'] = test1_['ConfirmedCases']*0.5 + test1_['ConfirmedCases_log']*0.5

test1_['Fatalities'] = test1_['Fatalities']*0.5 + test1_['Fatalities_log']*0.5
test1_.loc[test1.Country_Region_=='Ukraine',cols]
#df =test1[['ForecastId','ConfirmedCases','Fatalities']]
df =test1_[['ForecastId','ConfirmedCases','Fatalities']]

df.to_csv('submission.csv', index=False)
df.head()