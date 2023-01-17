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
train_df = pd.read_csv("../input/covid19-global-forecasting-week-3/train.csv")

test_df = pd.read_csv("../input/covid19-global-forecasting-week-3/test.csv")
train_df.info()
'''

# Replacing all the Province_State that are null by the Country_Region values

train_df.Province_State.fillna(train_df.Country_Region, inplace=True)

test_df.Province_State.fillna(test_df.Country_Region, inplace=True)

'''



# Replacing all the Province_State that are null by the Country_Region values

train_df.Province_State.fillna("unknown", inplace=True)

test_df.Province_State.fillna("unknown", inplace=True)



# Handling the Date column

# 1. Converting the object type column into datetime type

train_df.Date = train_df.Date.apply(pd.to_datetime)

test_df.Date = test_df.Date.apply(pd.to_datetime)



# 2. Creating new features

#train_df['ReportDay_year'] = train_df['Date'].dt.year #Not required this column because all the data is of this year

train_df['ReportDay_month'] = train_df['Date'].dt.month

train_df['ReportDay_week'] = train_df['Date'].dt.week

train_df['ReportDay_day'] = train_df['Date'].dt.day



#test_df['ReportDay_year'] = test_df['Date'].dt.year

test_df['ReportDay_month'] = test_df['Date'].dt.month

test_df['ReportDay_week'] = test_df['Date'].dt.week

test_df['ReportDay_day'] = test_df['Date'].dt.day
#Dropping the date column

train_df.drop("Date", inplace = True, axis = 1)

test_df.drop("Date", inplace = True, axis = 1)
'''

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()



train_df.Country_Region = le.fit_transform(train_df.Country_Region)

train_df['Province_State'] = le.fit_transform(train_df['Province_State'])



test_df.Country_Region = le.fit_transform(test_df.Country_Region)

test_df['Province_State'] = le.fit_transform(test_df['Province_State'])

'''


def one_hot(df, cols):

    """

    @param df pandas DataFrame

    @param cols a list of columns to encode 

    @return a DataFrame with one-hot encoding

    """

    i = 0

    for each in cols:

        #print (each)

        dummies = pd.get_dummies(df[each], prefix=each, drop_first= True)

        if i == 0: 

            print (dummies)

            i = i + 1

        df = pd.concat([df, dummies], axis=1)

    return df


#Handling categorical data



objList = train_df.select_dtypes(include = "object").columns

train_df = one_hot(train_df, objList) 

test_df = one_hot(test_df, objList) 



print (train_df.shape)


# Removing duplicate entries

train_df = train_df.loc[:,~train_df.columns.duplicated()]

test_df = test_df.loc[:,~test_df.columns.duplicated()]

print (test_df.shape)
# Dropping the object type columns

train_df.drop(objList, axis=1, inplace=True)

test_df.drop(objList, axis=1, inplace=True)

print (train_df.shape)

test_df.info()
X_train = train_df.drop(["Id", "ConfirmedCases", "Fatalities"], axis = 1)

#X_train_Fat = train_df.drop(["Id", "Fatalities"], axis = 1)



#X_train = train_df.loc[:, 'ForecastId', 'ForecastId']

Y_train_CC = train_df["ConfirmedCases"] 

Y_train_Fat = train_df["Fatalities"] 





#X_test = test_df.drop(["ForecastId"], axis = 1)

X_test = test_df.drop(["ForecastId"], axis = 1) 
from sklearn.model_selection import cross_val_score, KFold

from sklearn.metrics import make_scorer, r2_score, mean_squared_log_error



n_folds = 5

cv = KFold(n_splits = 10, shuffle=True, random_state=42).get_n_splits(X_train.values)



def test_model(model, colName):   

    msle = make_scorer(mean_squared_log_error)

    if colName == "CC":

        #print ("In")

        rmsle = np.sqrt(cross_val_score(model, X_train, Y_train_CC, cv=cv, scoring = msle))

    elif colName == "Fat": 

        rmsle = np.sqrt(cross_val_score(model, X_train, Y_train_Fat, cv=cv, scoring = msle))

    #print (rmsle)

    score_rmsle = [rmsle.mean()]

    return score_rmsle



def test_model_r2(model, colName):

    r2 = make_scorer(r2_score)

    if colName == "CC":

        r2_error = cross_val_score(model, X_train, Y_train_CC, cv=cv, scoring = r2)

    elif colName == "Fat": 

        r2_error = cross_val_score(model, X_train, Y_train_Fat, cv=cv, scoring = r2)

    score_r2 = [r2_error.mean()]

    return score_r2
'''

# BaggingRegressor

from sklearn.ensemble import BaggingRegressor

from sklearn.tree import DecisionTreeRegressor



clf_bgr_CC = BaggingRegressor(base_estimator = DecisionTreeRegressor())

clf_bgr_Fat = BaggingRegressor(base_estimator = DecisionTreeRegressor())



rmsle_bgr_CC = test_model(clf_bgr_CC, "CC")

rmsle_bgr_Fat = test_model(clf_bgr_Fat, "Fat")



print (rmsle_bgr_CC, rmsle_bgr_Fat)

'''


# XGBoost Regressor

import xgboost as xgb



clf_xgb_CC = xgb.XGBRegressor(n_estimators = 1250 , random_state = 0)

clf_xgb_Fat = xgb.XGBRegressor(n_estimators = 1250 , random_state = 0)



#rmsle_bgr_CC = clf_xgb_CC(clf_bgr_CC, "CC")

#rmsle_bgr_Fat = clf_xgb_Fat(clf_bgr_Fat, "Fat")



#print (rmsle_bgr_CC, rmsle_bgr_Fat)

clf_xgb_CC.fit(X_train, Y_train_CC)

Y_pred_CC = clf_xgb_CC.predict(X_test) 



clf_xgb_Fat.fit(X_train, Y_train_Fat)

Y_pred_Fat = clf_xgb_Fat.predict(X_test) 
print (Y_pred_Fat)
df_out = pd.DataFrame({'ForecastId': [], 'ConfirmedCases': [], 'Fatalities': []})

soln = pd.DataFrame({'ForecastId': test_df.ForecastId, 'ConfirmedCases': Y_pred_CC, 'Fatalities': Y_pred_Fat})

df_out = pd.concat([df_out, soln], axis=0)

df_out.ForecastId = df_out.ForecastId.astype('int')
df_out.to_csv('submission.csv', index=False)

print("Your submission was successfully saved!")