#import libs

import numpy as np

import pandas as pd



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

#df_train = pd.read_csv("train.csv")

#df_test = pd.read_csv("test.csv")



# Load Data

df_train = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-3/train.csv')

df_test = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-3/test.csv')

#pandas profiling over data to check for NaNs etc

import pandas_profiling as pp



pp.ProfileReport(df_train)

#fix data based off of pandas profiling report

def fillState(state, country):

    if state == "NA": return country

    return state



def fixData(input_set):

    input_set['Province_State'].fillna("NA", inplace=True)

    input_set['Province_State'] = input_set.loc[:, ['Province_State', 'Country_Region']].apply(lambda x : fillState(x['Province_State'], x['Country_Region']), axis=1)



    input_set['Date'] = pd.to_datetime(input_set['Date'], infer_datetime_format=True)

    input_set.loc[:, 'Date'] = input_set.Date.dt.strftime("%m%d")

    input_set["Date"]  = input_set["Date"].astype(int)

    return input_set

#prep data for train

X_train = df_train

X_test = df_test

#y1_train = df_train.iloc[:, -2]

#y2_train = df_train.iloc[:, -1]





X_train = fixData(X_train)

X_test = fixData(X_test)



X_train.head()
#fit_data

#from sklearn import preprocessing



#label_encoder = preprocessing.LabelEncoder()



#X_train.Country_Region = label_encoder.fit_transform(X_train.Country_Region)

#X_train['Country_Region'] = label_encoder.fit_transform(X_train['Province_State'])



#X_train.head()



#X_test.Country_Region = label_encoder.fit_transform(X_test.Country_Region)

#X_test['Country_Region'] = label_encoder.fit_transform(X_test['Province_State'])



#X_test.head()



#X_train.head()

#X_train.loc[X_train.Country_Region == 'Afghanistan', :]

#X_test.tail()

from sklearn import preprocessing



label_encoder = preprocessing.LabelEncoder()



from xgboost import XGBRegressor



countries = X_test.Country_Region.unique()
#Predict data and Create submission file from test data

sub = pd.DataFrame({'ForecastId': [], 'ConfirmedCases': [], 'Fatalities': []})



sub = []

for country in countries:

    province_list = X_train.loc[X_train['Country_Region'] == country].Province_State.unique()

    for province in province_list:

        X_train2 = X_train.loc[(X_train['Country_Region'] == country) & (X_train['Province_State'] == province),['Date']].astype('int')

        Y_train21 = X_train.loc[(X_train['Country_Region'] == country) & (X_train['Province_State'] == province),['ConfirmedCases']]

        Y_train22 = X_train.loc[(X_train['Country_Region'] == country) & (X_train['Province_State'] == province),['Fatalities']]

        X_test2 = X_test.loc[(X_test['Country_Region'] == country) & (X_test['Province_State'] == province), ['Date']].astype('int')

        X_forecastId2 = X_test.loc[(X_test['Country_Region'] == country) & (X_test['Province_State'] == province), ['ForecastId']]

        X_forecastId2 = X_forecastId2.values.tolist()

        X_forecastId2 = [v[0] for v in X_forecastId2]

        model2 = XGBRegressor(n_estimators=1020)

        model2.fit(X_train2, Y_train21)

        Y_pred2 = model2.predict(X_test2)

        model3 = XGBRegressor(n_estimators=1020)

        model3.fit(X_train2, Y_train22)

        Y_pred3 = model3.predict(X_test2)

        for j in range(len(Y_pred2)):

            dic = { 'ForecastId': X_forecastId2[j], 'ConfirmedCases': Y_pred2[j], 'Fatalities': Y_pred3[j]}

            sub.append(dic)

        
#submission.ForecastId = sub.ForecastId.astype('int')

#submission.to_csv('submission.csv', index=False)

submission = pd.DataFrame(sub)

submission[['ForecastId','ConfirmedCases','Fatalities']].to_csv(path_or_buf='submission.csv',index=False)
#from xgboost import plot_importance

#import matplotlib.pyplot as plt



# plot feature importance

#plot_importance(model2)

#pyplot.show()
"""

from sklearn.model_selection import train_test_split

# split data into train and test sets

seed = 7

test_size = 0.1

X_train, X_test, y_train, y_test = train_test_split(X_train, y1_train, test_size=test_size, random_state=seed)





y1_train = df_train.iloc[:, -2]

y2_train = df_train.iloc[:, -1]



#import XGBoost

from xgboost import XGBRegressor

from sklearn.metrics import accuracy_score



#create model and train

model = XGBRegressor(learning_rate = 0.05, n_estimators=1000, max_depth=5)

model.fit(X_train_CS, y1_X_train_CS)

"""

"""

# make predictions for test set

y_pred = model.predict(X_test)

predictions = [round(value) for value in y_pred]



accuracy = accuracy_score(y_test, predictions)

print("Accuracy: %.2f%%" % (accuracy * 100.0))

"""