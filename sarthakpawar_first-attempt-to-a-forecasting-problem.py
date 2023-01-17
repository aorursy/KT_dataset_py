#Load required Libraries

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

sns.set(context="notebook", style="darkgrid", palette="deep", font="sans-serif", font_scale=1, color_codes=True)

#Load Data and remove hyphen from Date column after convert the column to int

data= pd.read_csv("../input/covid19-global-forecasting-week-1/train.csv")

test = pd.read_csv("../input/covid19-global-forecasting-week-1/test.csv")  
data.head()
data['Province/State']=data['Province/State'].fillna('')
test['Province/State']=test['Province/State'].fillna('')
data['Province/State'].value_counts()
test['Province/State'].value_counts()
set(data['Province/State']).difference(set(test['Province/State']))
from sklearn.preprocessing import LabelEncoder
le_state=LabelEncoder()
le_state=le_state.fit(data['Province/State'])
data['Province/State']=le_state.transform(data['Province/State'])

test['Province/State']=le_state.transform(test['Province/State'])
set(test['Country/Region']).difference(set(data['Country/Region']))
le_country=LabelEncoder()
le_country=le_country.fit(data['Country/Region'])
data['Country/Region']=le_country.transform(data['Country/Region'])

test['Country/Region']=le_country.transform(test['Country/Region'])
data.head()
from datetime import datetime



datetime_str = '01/22/20 00:00:00'



datetime_object = datetime.strptime(datetime_str, '%m/%d/%y %H:%M:%S')

data['days']=pd.to_datetime(data['Date']).sub(datetime_object)/np.timedelta64(1, 'D')
test['days']=pd.to_datetime(test['Date']).sub(datetime_object)/np.timedelta64(1, 'D')
import xgboost as xgb
data.head()
data.isna().sum(axis=0)/data.shape[0]
test.isna().sum(axis=0)
#Asign columns for training and testing

x = data[['Province/State','Country/Region','Lat', 'Long', 'days']]

y1 = data['ConfirmedCases']

y2 = data['Fatalities']

x_test_set = test[['Province/State','Country/Region','Lat', 'Long', 'days']]

#y_test = test[['ConfirmedCases']]
import math



#A function to calculate Root Mean Squared Logarithmic Error (RMSLE)

def rmsle(y, y_pred):

    assert len(y) == len(y_pred)

    terms_to_sum = [(math.log(y_pred[i] + 1) - math.log(y[i] + 1)) ** 2.0 for i,pred in enumerate(y_pred)]

    return (sum(terms_to_sum) * (1.0/len(y))) ** 0.5
model=xgb.XGBRegressor(base_score=0.5, booster=None, colsample_bylevel=1,

             colsample_bynode=1, colsample_bytree=1, gamma=0.1, gpu_id=-1,

             importance_type='gain', interaction_constraints=None,

             learning_rate=0.1, max_delta_step=0, max_depth=20,

             min_child_weight=1,  monotone_constraints=None,

             n_estimators=2048, n_jobs=0, num_parallel_tree=10,

             objective='reg:squarederror', random_state=0, reg_alpha=0,

             reg_lambda=1, scale_pos_weight=1, subsample=1, tree_method=None,

             validate_parameters=False, verbosity=None)
model.fit(x,y1)
# rmsle(y_test['ConfirmedCases'].to_list(),pred1)
pred1 = model.predict(x_test_set)
pred1=np.array([int(round(p)) for p in pred1])
pred1[np.array(pred1)<0]=0
##

model.fit(x,y2)

pred2 = model.predict(x_test_set)

pred2=np.array([int(round(p)) for p in pred2])

pred2[pred2<0]=0
Sub = pd.read_csv("../input/covid19-global-forecasting-week-1/submission.csv")

Sub.columns

sub_new = Sub["ForecastId"]
OP = pd.DataFrame()

OP['ForecastId']=Sub["ForecastId"]

OP['ConfirmedCases']=pred1

OP['Fatalities']=pred2

# OP.columns = ['ConfirmedCases', 'Fatalities', 'ForecastId']

# OP = OP[['ForecastId','ConfirmedCases', 'Fatalities']]

OP["ConfirmedCases"] = OP["ConfirmedCases"].astype(int)

OP["Fatalities"] = OP["Fatalities"].astype(int)

OP.head()
OP.to_csv("submission.csv",index=False)