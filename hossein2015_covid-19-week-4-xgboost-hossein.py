import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import nltk

from sklearn.preprocessing import LabelBinarizer,LabelEncoder,StandardScaler,MinMaxScaler

from sklearn.linear_model import LogisticRegression,SGDClassifier,LinearRegression

from sklearn.naive_bayes import MultinomialNB

from sklearn.svm import SVC

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import classification_report,confusion_matrix,accuracy_score

from sklearn.model_selection import train_test_split

import keras

from keras.wrappers.scikit_learn import KerasRegressor

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import KFold

from keras.models import Sequential

from keras.layers import Dense,LSTM

import tensorflow as tf
train_df = pd.read_csv("../input/covid19-global-forecasting-week-4/train.csv")

test_df = pd.read_csv("../input/covid19-global-forecasting-week-4/test.csv")

train_df['Province_State'].fillna("",inplace = True)

test_df['Province_State'].fillna("",inplace = True)

train_df['Country_Region'] = train_df['Country_Region'] + ' ' + train_df['Province_State']

test_df['Country_Region'] = test_df['Country_Region'] + ' ' + test_df['Province_State']

del train_df['Province_State']

del test_df['Province_State']

def split_date(date):

    date = date.split('-')

    date[0] = int(date[0])

    if(date[1][0] == '0'):

        date[1] = int(date[1][1])

    else:

        date[1] = int(date[1])

    if(date[2][0] == '0'):

        date[2] = int(date[2][1])

    else:

        date[2] = int(date[2])    

    return date

train_df.Date = train_df.Date.apply(split_date)

test_df.Date = test_df.Date.apply(split_date)
year = []

month = []

day = []

for i in train_df.Date:

    year.append(i[0])

    month.append(i[1])

    day.append(i[2])

train_df['Year'] = year

train_df['Month'] = month

train_df['Day'] = day

del train_df['Date']

year = []

month = []

day = []

for i in test_df.Date:

    year.append(i[0])

    month.append(i[1])

    day.append(i[2])

test_df['Year'] = year

test_df['Month'] = month

test_df['Day'] = day

del test_df['Date']

del train_df['Id']

del test_df['ForecastId']

del train_df['Year']

del test_df['Year']

train_df['ConfirmedCases'] = train_df['ConfirmedCases'].apply(int)

train_df['Fatalities'] = train_df['Fatalities'].apply(int)

cases = train_df.ConfirmedCases

fatalities = train_df.Fatalities

del train_df['ConfirmedCases']

del train_df['Fatalities']

lb = LabelEncoder()

train_df['Country_Region'] = lb.fit_transform(train_df['Country_Region'])

test_df['Country_Region'] = lb.transform(test_df['Country_Region'])

scaler = MinMaxScaler()

x_train = scaler.fit_transform(train_df.values)

x_test = scaler.transform(test_df.values)
from xgboost import XGBRegressor

rf = XGBRegressor(n_estimators = 1500 , random_state = 0 , max_depth = 15)

rf.fit(x_train,cases)

cases_pred = rf.predict(x_test)

cases_pred = np.around(cases_pred,decimals = 0)

x_train_cas = []

for i in range(len(x_train)):

    x = list(x_train[i])

    x.append(cases[i])

    x_train_cas.append(x)

x_train_cas[0]

x_train_cas = np.array(x_train_cas)

rf = XGBRegressor(n_estimators = 1500 , random_state = 0 , max_depth = 15)

rf.fit(x_train_cas,fatalities)

x_test_cas = []

for i in range(len(x_test)):

    x = list(x_test[i])

    x.append(cases_pred[i])

    x_test_cas.append(x)

x_test_cas[0]

x_test_cas = np.array(x_test_cas)

fatalities_pred = rf.predict(x_test_cas)

fatalities_pred = np.around(fatalities_pred,decimals = 0)
submission = pd.read_csv("../input/covid19-global-forecasting-week-4/submission.csv")

submission['ConfirmedCases'] = cases_pred

submission['Fatalities'] = fatalities_pred

submission.to_csv("submission.csv" , index = False)
#pip install dabl
#import libraries#

#import numpy as np 

#import pandas as pd 

#import seaborn as sns

#import missingno as msno

#import dabl

#from sklearn.model_selection import train_test_split

#from sklearn.pipeline import Pipeline

#from xgboost import XGBRegressor

#from sklearn.multioutput import MultiOutputRegressor

#from sklearn.impute import SimpleImputer

#from sklearn.preprocessing import LabelEncoder
#Load Train & Test Data#

#train = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/train.csv')

#test = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/test.csv')

#train.shape()

#train.head(5)

#test.head(5)

#test.shape()

#display(train.head())

#display(train.describe())
#Converting 'Date' variable from string type to DateTime type (Train & Test)#

#train['Date'] = pd.to_datetime(train['Date'], infer_datetime_format=True)

#test['Date'] = pd.to_datetime(test['Date'], infer_datetime_format=True)

#train.loc[:, 'Date'] = train.Date.dt.strftime('%y%m%d')

#train.loc[:, 'Date'] = train['Date'].astype(int)

#test.loc[:, 'Date'] = test.Date.dt.strftime('%y%m%d')

#test.loc[:, 'Date'] = test['Date'].astype(int)

#train_dmin=train['Date'].min()

#print('Minimum date from training set: {}'.format(train_dmin))

#train_dmax=train['Date'].max()

#print('Minimum date from training set: {}'.format(train_dmax))

#test_dmin=test['Date'].min()

#print('Minimum date from training set: {}'.format(test_dmin))

#test_dmax=train['Date'].max()

#print('Minimum date from training set: {}'.format(test_dmax))

#train.head(5)
#Replace NAN cells for the Province_State variable with the ones correpons to the Country_Region variable#

#train['Province_State'] = np.where(train['Province_State'] == 'nan',train['Country_Region'],train['Province_State'])

#test['Province_State'] = np.where(test['Province_State'] == 'nan',test['Country_Region'],test['Province_State'])

#convert_dict = {'Province_State': str}

#train = train.astype(convert_dict)

#test = test.astype(convert_dict)
#train_clean=dabl.clean(train,verbose=1)

#types=dabl.detect_types(train)

#print(types)

#train.info

#msno.bar(train)
#Label Encoding for the categorical variables#

#label_encoder1 = LabelEncoder()

#label_encoder2 = LabelEncoder()

#train['Province_State'] = label_encoder1.fit_transform(train['Province_State'])

#test['Province_State'] = label_encoder1.transform(test['Province_State'])

#train['Country_Region'] = label_encoder2.fit_transform(train['Country_Region'])

#test['Country_Region'] = label_encoder2.transform(test['Country_Region'])
#Bar Plot#

#sns.countplot(y="Country_Region", data=train,order=train["Country_Region"].value_counts(ascending=False).iloc[:10].index)
#Linear Regression Visualization#

#sns.regplot(x=train["ConfirmedCases"], y=train["Fatalities"], fit_reg=True)
#For the output file#

#Test_id = test.ForecastId

#train.drop(['Id'], axis=1, inplace=True)

#test.drop('ForecastId', axis=1, inplace=True)
#train['Date'] = pd.to_datetime(train['Date'], infer_datetime_format=True)

#test['Date'] = pd.to_datetime(test['Date'], infer_datetime_format=True)

#train.loc[:, 'Date'] = train.Date.dt.strftime('%y%m%d')

#train.loc[:, 'Date'] = train['Date'].astype(int)

#test.loc[:, 'Date'] = test.Date.dt.strftime('%y%m%d')

#test.loc[:, 'Date'] = test['Date'].astype(int)

#Variables Allocation in the training & test data sets#

#X_train = train[['Province_State','Country_Region','Date']]

#y_train = train[['ConfirmedCases', 'Fatalities']]

#y_train_confirm = y_train.ConfirmedCases

#y_train_fatality = y_train.Fatalities
#model1 = XGBRegressor(n_estimators=40000)

#model1.fit(X_train, y_train_confirm)

#y_pred_confirm = model1.predict(test)
#model2 = XGBRegressor(n_estimators=20000)

#model2.fit(X_train, y_train_fatality)

#y_pred_fat = model2.predict(test)
#df_sub = pd.DataFrame()

#df_sub['ForecastId'] = Test_id

#df_sub['ConfirmedCases'] = y_pred_confirm

#df_sub['Fatalities'] = y_pred_fat

#df_sub.to_csv('submission.csv', index=False)