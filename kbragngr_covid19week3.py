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
train = pd.read_csv("../input/covid19-3/train_covid19.csv",parse_dates=['Date'])

test = pd.read_csv("../input/covid19-3/test_covid19.csv",parse_dates=['Date'])

submission = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-3/submission.csv")

train = train.drop(['Unnamed: 0','Id'],axis=1)

test = test.drop(["Unnamed: 0","ForecastId"],axis=1)
train.head()
test.isnull().any().sum
train["days"] = pd.to_datetime(train["Date"]).sub(pd.Timestamp('2020-01-21')).dt.days

test["days"] = pd.to_datetime(test["Date"]).sub(pd.Timestamp('2020-01-21')).dt.days
"""

train["Country_Province"] = train["Country_Region"] + "/" +train["Province_State"].fillna("0")

test["Country_Province"] = test["Country_Region"] + "/" + test["Province_State"].fillna("0")

 

train.head()



"""
train["Province_State"].fillna("unkown",inplace=True)

train["Cigarette_Consumption"].fillna(train["Cigarette_Consumption"].mean(),inplace=True)

train["Life_expectancy"].fillna(train["Life_expectancy"].mean(),inplace=True)

train["age_0_14"].fillna(train["age_0_14"].mean(),inplace=True)

train["age_15_64"].fillna(train["age_15_64"].mean(),inplace=True)

train["age_over_65"].fillna(train["age_over_65"].mean(),inplace=True)





test["Province_State"].fillna("unkown",inplace=True)

test["Cigarette_Consumption"].fillna(test["Cigarette_Consumption"].mean(),inplace=True)

test["Life_expectancy"].fillna(test["Life_expectancy"].mean(),inplace=True)

test["age_0_14"].fillna(test["age_0_14"].mean(),inplace=True)

test["age_15_64"].fillna(test["age_15_64"].mean(),inplace=True)

test["age_over_65"].fillna(test["age_over_65"].mean(),inplace=True)





"""



train["Cigarette_consumption_mean"] = (train["Cigarette_Consumption"]/(train["Cigarette_Consumption"].sum()))*100.0



test["Cigarette_consumption_mean"] = (test["Cigarette_Consumption"]/(test["Cigarette_Consumption"].sum()))*100.0"""
#train["Cigarette_consumption_mean"].sum()
train = train.drop(["Date"],axis=1)

test = test.drop(["Date"],axis=1)

train.isnull().any().sum
from sklearn.preprocessing import LabelEncoder



cat_features = ["Country_Region","Province_State"]

encoder = LabelEncoder()



train_encoded = train[cat_features].apply(encoder.fit_transform)

test_encoded = test[cat_features].apply(encoder.fit_transform)

from sklearn import preprocessing

def normalize(df):

    x = df.values 

    min_max_scaler = preprocessing.MinMaxScaler()

    x_scaled = min_max_scaler.fit_transform(x)

    return pd.DataFrame(x_scaled)
cols = ['Cigarette_Consumption','Life_expectancy','days']



X = train[cols].join(train_encoded)



TestX = test[cols].join(test_encoded)

X.head()



features = X.columns
"""

X = normalize(X)

TestX = normalize(TestX)

"""


y_Confirmed = train["ConfirmedCases"].to_numpy()

y_Fatalities = train["Fatalities"].to_numpy()



y_Confirmed = y_Confirmed.reshape(-1)

y_Fatalities = y_Fatalities.reshape(-1)
from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import  GridSearchCV



def RandomForestReg(X_train,y_train):

    rf_model = RandomForestRegressor(random_state = 42)

    

    rf_model.fit(X_train, y_train)

    return rf_model

    

    

ConfirmedModel = RandomForestReg(X, y_Confirmed)

FatalitiesModel = RandomForestReg(X,y_Fatalities)
submission["ConfirmedCases"] = ConfirmedModel.predict(TestX)

submission["Fatalities"] = FatalitiesModel.predict(TestX)

submission[:20]
submission.tail()
ImportanceC = pd.DataFrame({"Importance": ConfirmedModel.feature_importances_*100},

                         index = X.columns)
import matplotlib.pyplot as plt

ImportanceC.sort_values(by = "Importance", 

                       axis = 0, 

                       ascending = True).plot(kind ="barh", color = "r",figsize=(12,12))



plt.xlabel("Confirmed Cases: Feature importances")
ImportanceF = pd.DataFrame({"Importance": FatalitiesModel.feature_importances_*100},

                         index = X.columns)
ImportanceF.sort_values(by = "Importance", 

                       axis = 0, 

                       ascending = True).plot(kind ="barh", color = "r",figsize=(12,12))



plt.xlabel("Fatalities feature importances")
submission.to_csv("submission.csv",index=False)