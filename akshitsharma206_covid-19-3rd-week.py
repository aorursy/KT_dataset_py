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
import pandas as pd

import numpy as np 

import seaborn as sns

import matplotlib.pyplot as plt

import pandas as pd

from datetime import datetime

from sklearn import preprocessing

from xgboost import XGBRegressor

from sklearn.tree import DecisionTreeRegressor

from lightgbm import LGBMRegressor

from catboost import CatBoostRegressor

from sklearn.linear_model import LogisticRegression

from sklearn.linear_model import LinearRegression

from sklearn.ensemble import RandomForestRegressor 
path = '/kaggle/input/covid19-global-forecasting-week-3/'

train = pd.read_csv(path+'train.csv')

test = pd.read_csv(path+'test.csv')
#Displaying the first 5 rows of the dataset

train.head()
#Displaying first 5 rows of the dataset

test.head()
#Checking the null values in train-data

train.isnull().sum()
#Checking the Null values in test-data

test.isnull().sum()
#importing some extra libraries

%matplotlib inline

import seaborn as sns

import plotly.express as px

import plotly.graph_objects as go

import matplotlib.dates as mdates

#from fbprophet import Prophet
#Checking the affected cases

case = train.groupby('Country_Region')['ConfirmedCases'].max().sort_values(ascending=False).to_frame()

case.style.background_gradient(cmap='GnBu')
#Taking the top 10 countries

case=case.reset_index().head(10)
#Plotting a bar chart for top 10 countries

plt.figure(figsize =(10,10))

sns.set_color_codes("muted")

chart=sns.barplot(x='Country_Region',y='ConfirmedCases',data=case,label="Total Number of Cases")

for item in chart.get_xticklabels():

    item.set_rotation(90)
#making scatterplot using plotly

fig = go.Figure(data=go.Scatter(x=case['Country_Region'], y=case['ConfirmedCases'],

mode='lines+markers',hovertemplate = "Cases: %{y}<br> %{x}<extra></extra>",showlegend = False))

fig.show()
#Basic Descreptive Statistics

train.loc[:, ['ConfirmedCases', 'Fatalities']].describe()
#Converting datetime to integer

train["Date"] = train["Date"].apply(lambda x: x.replace("-",""))

train["Date"]  = train["Date"].astype(int)

test["Date"] = test["Date"].apply(lambda x: x.replace("-",""))

test["Date"]  = test["Date"].astype(int)
#Makinga a function for Null Values

EMPTY_VAL = "EMPTY_VAL"

def fillState(state, country):

    if state == EMPTY_VAL: return country

    return state
train['Province_State'].fillna(EMPTY_VAL, inplace=True)

train['Province_State'] = train.loc[:, ['Province_State', 'Country_Region']].apply(lambda x : fillState(x['Province_State'], x['Country_Region']), axis=1)

test['Province_State'].fillna(EMPTY_VAL, inplace=True)

test['Province_State'] = test.loc[:, ['Province_State', 'Country_Region']].apply(lambda x : fillState(x['Province_State'], x['Country_Region']), axis=1)
from sklearn.preprocessing import LabelEncoder



le = LabelEncoder()



train['Country_Region'] = le.fit_transform(train['Country_Region'])



train['Province_State'] = le.fit_transform(train['Province_State'])



test['Country_Region'] = le.fit_transform(test['Country_Region'])



test['Province_State'] = le.fit_transform(test['Province_State'])
#taking unique countries only

countries = train['Country_Region'].unique()
def build_model_1():

    model = RandomForestRegressor(n_estimators = 100, random_state = 0) 

    return model



def build_model_2():

    model = XGBRegressor(n_estimators=1000)

    return model



def build_model_3():

    model = DecisionTreeRegressor(random_state=1)

    return model



def build_model_4():

    model = LogisticRegression()

    return model



def build_model_5():

    model = LinearRegression()

    return model



def build_model_6():

    model = LGBMRegressor(random_state=5)

    return model



def build_model_7():

    model = LGBMRegressor(iterations=2)

    return model
out = pd.DataFrame({'ForecastId': [], 'ConfirmedCases': [], 'Fatalities': []})
for country in range(len(countries)):

    country_train = train.loc[train['Country_Region'] == countries[country]]

    country_test = test.loc[test['Country_Region'] == countries[country]]

    print("Current Country: ", countries[country])

    

#     print(current_country_df_train.head(2))

#     print(current_country_df_test.head(2))

    

    # Create X and Y

    xtrain = country_train[['Country_Region', 'Province_State', 'Date']].to_numpy()

    y1train = country_train[['ConfirmedCases']].to_numpy()

    y2train = country_train[['Fatalities']].to_numpy()

    xtest = country_test[['Country_Region', 'Province_State', 'Date']].to_numpy()

    

    y1train = y1train.reshape(-1)

    y2train = y2train.reshape(-1)

#   print(X_train.shape, Y1_train.shape, Y2_train.shape)

#   print(X_train.shape, Y1_train.reshape(-1), Y2_train.shape)

    

    

    model1 = build_model_2()

    model1.fit(xtrain, y1train)

    res_cnf_cls = model1.predict(xtest)

    

    

    model2 = build_model_2()

    model2.fit(xtrain, y2train)

    res_fac = model2.predict(xtest)

    

    country_test_Id = country_test.loc[:, 'ForecastId']

    country_test_Id = country_test_Id.astype(int)

    

    ans = pd.DataFrame({'ForecastId': country_test_Id, 'ConfirmedCases': res_cnf_cls, 'Fatalities': res_fac})

    out = pd.concat([out, ans], axis=0)

   

    

#     if country == 1:

#         break
out["ForecastId"] = out["ForecastId"].astype(int)
out.to_csv('submission.csv', index=False)