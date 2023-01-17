# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
EMPTY_VAL = "EMPTY_VAL"

NAN_VAL = "NaN"



def get_location(county, state, country):

    if county == EMPTY_VAL:

        if state == EMPTY_VAL:

            return country

        else:

            return country + '_' + state

    else:

        return country + '_' + state + '_' + county

    return country + '_' + state + '_' + county
PATH_WEEK_5='/kaggle/input/covid19-global-forecasting-week-5'

#PATH_WEEK_5='./data/covid19-global-forecasting-week-5'

df_train = pd.read_csv(f'{PATH_WEEK_5}/train.csv')

df_test = pd.read_csv(f'{PATH_WEEK_5}/test.csv')
X_Train = df_train.copy()



#Unite  'County', 'Province_State' and 'Country_Region'

X_Train['County'].fillna(EMPTY_VAL, inplace=True)

X_Train['Province_State'].fillna(EMPTY_VAL, inplace=True)

X_Train['Location'] = X_Train[['County', 'Province_State', 'Country_Region']].apply(lambda x : get_location(x['County'], x['Province_State'], x['Country_Region']), axis=1)

#X_Train['Mortality'] = X_Train[['County', 'Province_State', 'Country_Region']].apply(lambda x : get_location(x['County'], x['Province_State'], x['Country_Region']), axis=1)



# Drop 'County' and 'Province_State' for clearence

#X_Train.drop("County",axis=1,inplace=True)

#X_Train.drop("Province_State",axis=1,inplace=True)

#X_Train.drop("Country_Region",axis=1,inplace=True)



#X_Train["Date"] = pd.to_datetime(X_Train["Date"]).dt.strftime("%Y%m%d")

X_Train["Date"] = pd.to_datetime(X_Train["Date"]).dt.strftime("%m%d")

X_Train["Date"]  = X_Train["Date"].astype(int)



X_Train['TargetValue'][X_Train["TargetValue"] < 0] = 0
X_Pred = df_test.copy()



#Unite  'County', 'Province_State' and 'Country_Region'

X_Pred['County'].fillna(EMPTY_VAL, inplace=True)

X_Pred['Province_State'].fillna(EMPTY_VAL, inplace=True)

X_Pred['Location'] = X_Pred[['County', 'Province_State', 'Country_Region']].apply(lambda x : get_location(x['County'], x['Province_State'], x['Country_Region']), axis=1)



# Drop 'County' and 'Province_State' for clearence

#X_Pred.drop("County",axis=1,inplace=True)

#X_Pred.drop("Province_State",axis=1,inplace=True)

#X_Pred.drop("Country_Region",axis=1,inplace=True)



#X_Pred["Date"] = pd.to_datetime(X_Pred["Date"]).dt.strftime("%Y%m%d")

X_Pred["Date"] = pd.to_datetime(X_Pred["Date"]).dt.strftime("%m%d")

X_Pred["Date"]  = X_Pred["Date"].astype(int)

X_Pred.drop(['ForecastId'],axis=1,inplace=True)

X_Pred.index.name = 'Id'

X_Pred.head()
from sklearn.preprocessing import LabelEncoder 

  

le = LabelEncoder() 

  

X_Train['Location']= le.fit_transform(X_Train['Location']) 

X_Train['County']= le.fit_transform(X_Train['County'])

X_Train['Province_State']= le.fit_transform(X_Train['Province_State'])

X_Train['Country_Region']= le.fit_transform(X_Train['Country_Region'])

X_Train['Target']= le.fit_transform(X_Train['Target']) 



X_Pred['Location']= le.fit_transform(X_Pred['Location']) 

X_Pred['County']= le.fit_transform(X_Pred['County'])

X_Pred['Province_State']= le.fit_transform(X_Pred['Province_State'])

X_Pred['Country_Region']= le.fit_transform(X_Pred['Country_Region'])

X_Pred['Target']= le.fit_transform(X_Pred['Target']) 
from sklearn.model_selection import train_test_split



predictors = X_Train.drop(['TargetValue', 'Id'], axis=1)

target = X_Train["TargetValue"]

df_X_train, df_X_test, df_y_train, df_y_test = train_test_split(predictors, target, test_size = 0.2#, random_state = 42

                                                               )
from sklearn.ensemble import RandomForestRegressor
rfr_fit = None;



rfr_fit = RandomForestRegressor(n_jobs=-1)

scores = []

rfr_fit.set_params(

    n_estimators = 100,

    random_state = 42,

    max_depth = 24,

    min_samples_leaf = 2

);



rfr_fit.fit(df_X_train, df_y_train)
from sklearn import metrics

from sklearn.metrics import mean_absolute_error



rfr_y_prediction = rfr_fit.predict(df_X_test)

val_mae = mean_absolute_error(rfr_y_prediction,df_y_test)

print(val_mae)
rfr_y_prediction = rfr_fit.predict(X_Pred)
y_pred = rfr_y_prediction
pred_list = [int(x) for x in y_pred]



output = pd.DataFrame({'Id': X_Pred.index, 'TargetValue': pred_list})

print(output)
q05 = output.groupby('Id')['TargetValue'].quantile(q=0.05).reset_index()

q50 = output.groupby('Id')['TargetValue'].quantile(q=0.5).reset_index()

q95 = output.groupby('Id')['TargetValue'].quantile(q=0.95).reset_index()



q05.columns=['Id','0.05']

q50.columns=['Id','0.5']

q95.columns=['Id','0.95']
concatDF = pd.concat([q05,q50['0.5'],q95['0.95']],1)

concatDF['Id'] = concatDF['Id'] + 1

#concatDF.head(10)
sub = pd.melt(concatDF, id_vars=['Id'], value_vars=['0.05','0.5','0.95'])

sub['ForecastId_Quantile']=sub['Id'].astype(str)+'_'+sub['variable']

sub['TargetValue']=sub['value']

sub=sub[['ForecastId_Quantile','TargetValue']]

sub.reset_index(drop=True,inplace=True)

sub.to_csv("submission.csv",index=False)

#sub.head(10)