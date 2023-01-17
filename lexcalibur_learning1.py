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
import pandas as pd                #Pandas library for data analysis

import numpy as np                 #For numerical analysis of data

import matplotlib.pyplot as plt    #Python's plotting 





import plotly.express as px       #Plotly for plotting the COVID-19 Spread.

import plotly.offline as py       #Plotly for plotting the COVID-19 Spread.

import seaborn as sns             #Seaborn for data plotting

import plotly.graph_objects as go #Plotlygo for plotting



from plotly.subplots import make_subplots





import glob                       #For assigning the path

import os                         #OS Library for implementing the functions.



import warnings

warnings.filterwarnings('ignore') 



#Selcting the other essential libraries for data manipulation

from sklearn import preprocessing

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import MinMaxScaler

import datetime as dt

import xgboost as xgb

from xgboost import XGBRegressor
#Importing the essential datasets from the challenge



training_data = pd.read_csv('../input/covid19-global-forecasting-week-4/train.csv')

testing_data = pd.read_csv('../input/covid19-global-forecasting-week-4/test.csv')
print(training_data.isnull().sum())

print(testing_data.isnull().sum())



print(training_data.dtypes)

print(testing_data.dtypes)



training_data['Province_State'].fillna("",inplace = True)

testing_data['Province_State'].fillna("",inplace = True)
#Merging the columns together



training_data['Country_Region'] = training_data['Country_Region'] + ' ' + training_data['Province_State']

testing_data['Country_Region'] = testing_data['Country_Region'] + ' ' + testing_data['Province_State']

del training_data['Province_State']

del testing_data['Province_State']



#Creating a function to split-date



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



training_data.Date = training_data.Date.apply(split_date)

testing_data.Date = testing_data.Date.apply(split_date)
#Manipulation of columns for both training dataset



year = []

month = []

day = []



for i in training_data.Date:

    year.append(i[0])

    month.append(i[1])

    day.append(i[2])

    

training_data['Year'] = year

training_data['Month'] = month

training_data['Day'] = day

del training_data['Date']



#Manipulation of columns for both testing dataset



year = []

month = []

day = []

for i in testing_data.Date:

    year.append(i[0])

    month.append(i[1])

    day.append(i[2])

    

testing_data['Year'] = year

testing_data['Month'] = month

testing_data['Day'] = day

del testing_data['Date']

del training_data['Id']

del testing_data['ForecastId']

del testing_data['Year']

del training_data['Year']
training_data
training_data['ConfirmedCases'] = training_data['ConfirmedCases'].apply(int)

training_data['Fatalities'] = training_data['Fatalities'].apply(int)



cases = training_data.ConfirmedCases

fatalities = training_data.Fatalities

del training_data['ConfirmedCases']

del training_data['Fatalities']



lb = LabelEncoder()

training_data['Country_Region'] = lb.fit_transform(training_data['Country_Region'])

testing_data['Country_Region'] = lb.transform(testing_data['Country_Region'])



scaler = MinMaxScaler()

x_train = scaler.fit_transform(training_data.values)

x_test = scaler.transform(testing_data.values)
from xgboost import XGBRegressor



rf = XGBRegressor(n_estimators = 1500 , max_depth = 15, learning_rate=0.1)

rf.fit(x_train,cases)

cases_pred = rf.predict(x_test)



rf = XGBRegressor(n_estimators = 1500 , max_depth = 15, learning_rate=0.1)

rf.fit(x_train,fatalities)

fatalities_pred = rf.predict(x_test)
#Rouding off the prediction values and converting negatives to zero

cases_pred = np.around(cases_pred)

fatalities_pred = np.around(fatalities_pred)



cases_pred[cases_pred < 0] = 0

fatalities_pred[fatalities_pred < 0] = 0
#Importing the dataset for generating output

submission_dataset = pd.read_csv('../input/covid19-global-forecasting-week-4/submission.csv')



#Adding results to the dataset

submission_dataset['ConfirmedCases'] = cases_pred

submission_dataset['Fatalities'] = fatalities_pred



submission_dataset.head(50)
#Submitting the dataset

submission_dataset.to_csv("submission.csv" , index = False)