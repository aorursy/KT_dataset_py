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
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score

from sklearn.model_selection import train_test_split

from xgboost import XGBRegressor

from sklearn.preprocessing import LabelBinarizer,LabelEncoder,StandardScaler,MinMaxScaler
train_df = pd.read_csv("../input/covid19-global-forecasting-week-3/train.csv")

test_df = pd.read_csv("../input/covid19-global-forecasting-week-3/test.csv")

submission_df = pd.read_csv("../input/covid19-global-forecasting-week-3/submission.csv")
train_df['Province_State'].fillna('',inplace=True)

test_df['Province_State'].fillna('',inplace=True)
lb = LabelEncoder()

train_df['Country_Region'] = lb.fit_transform(train_df['Country_Region'])

test_df['Country_Region'] = lb.transform(test_df['Country_Region'])



lb1 = LabelEncoder()

train_df['Province_State'] = lb.fit_transform(train_df['Province_State'])

test_df['Province_State'] = lb.transform(test_df['Province_State'])
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

train_df['ConfirmedCases'] = train_df['ConfirmedCases'].apply(int)

train_df['Fatalities'] = train_df['Fatalities'].apply(int)



cases = train_df.ConfirmedCases

fatalities = train_df.Fatalities

del train_df['ConfirmedCases']

del train_df['Fatalities']
scaler = MinMaxScaler()

x_train = scaler.fit_transform(train_df.values)

x_test = scaler.transform(test_df.values)
rf = XGBRegressor(n_estimators = 1500 , random_state = 0 , max_depth = 15)

rf.fit(x_train,cases)



cases_pred = rf.predict(x_test)

cases_pred = np.around(cases_pred,decimals = 0)

cases_pred



x_train_cas = []

for i in range(len(x_train)):

    x = list(x_train[i])

    x.append(cases[i])

    x_train_cas.append(x)



x_train_cas = np.array(x_train_cas)
rf = XGBRegressor(n_estimators = 1500 , random_state = 0 , max_depth = 15)

rf.fit(x_train_cas,fatalities)



x_test_cas = []

for i in range(len(x_test)):

    x = list(x_test[i])

    x.append(cases_pred[i])

    x_test_cas.append(x)



x_test_cas = np.array(x_test_cas)



fatalities_pred = rf.predict(x_test_cas)

fatalities_pred = np.around(fatalities_pred,decimals = 0)

fatalities_pred
submission_df['ConfirmedCases'] = cases_pred

submission_df['Fatalities'] = fatalities_pred
submission_df.to_csv("submission.csv" , index = False)