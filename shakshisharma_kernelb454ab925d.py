#Import Libararies

import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn import preprocessing

import time

from datetime import datetime 

import scipy.stats as stats

import statsmodels.api as sm
#  Read datasets

test = pd.read_csv("../input/covid19-global-forecasting-week-1/test.csv")

train = pd.read_csv("../input/covid19-global-forecasting-week-1/train.csv")
#We are using weather data provided on Kaggle

weather=pd.read_csv("../input/weather-data/training_data_with_weather_info.csv")
#We are using Tanu's dataset of population based on webscraping

population=pd.read_csv("../input/population3/population_by_country_2020.csv")
# Select required columns and rename few of them

population = population[['Country (or dependency)', 'Population (2020)', 'Density (P/Km²)', 'Land Area (Km²)', 'Med. Age', 'Urban Pop %']]

population.columns = ['Country (or dependency)', 'Population (2020)', 'Density', 'Land Area', 'Med Age', 'Urban Pop']
# Replace United States by US

population.loc[population['Country (or dependency)']=='United States', 'Country (or dependency)'] = 'US'
# Handling Urban Pop values

population['Urban Pop'] = population['Urban Pop'].str.rstrip('%')

p=population.loc[population['Urban Pop']!='N.A.', 'Urban Pop'].median()

population.loc[population['Urban Pop']=='N.A.', 'Urban Pop']= int(p)

population['Urban Pop'] = population['Urban Pop'].astype('int64')
# Handling Med Age values

population.loc[population['Med Age']=='N.A.', 'Med Age'] = int(population.loc[population['Med Age']!='N.A.', 'Med Age'].mode()[0])

population['Med Age'] = population['Med Age'].astype('int64')
print("Combined dataset")

corona_data = weather.merge(population, left_on='Country/Region', right_on='Country (or dependency)', how='left')

corona_data.shape
#checking for null values

sns.heatmap(corona_data.isnull(),yticklabels=False,cbar=False,cmap='viridis')
#Drop Province/State 

corona_data.drop('Province/State', axis=1, inplace=True)
#Drop Country or dependency

corona_data.drop('Country (or dependency)', axis=1, inplace=True)
#checking for null values

sns.heatmap(corona_data.isnull(),yticklabels=False,cbar=False,cmap='viridis')
corona_data[['Population (2020)', 'Density', 'Land Area', 'Med Age', 'Urban Pop']] = corona_data[['Population (2020)', 'Density', 'Land Area', 'Med Age', 'Urban Pop']].fillna(0)
#checking for null values

sns.heatmap(corona_data.isnull(),yticklabels=False,cbar=False,cmap='viridis')
from sklearn.preprocessing import LabelEncoder

labelencoder_X = LabelEncoder() 

corona_data.iloc[:, 1] = labelencoder_X.fit_transform(corona_data.iloc[:, 1])
corona_data['day']=pd.DatetimeIndex(corona_data['Date']).day

corona_data['year'] = pd.DatetimeIndex(corona_data['Date']).year

corona_data['month'] = pd.DatetimeIndex(corona_data['Date']).month

corona_data.head()
corona_data['Population (2020)'] = corona_data['Population (2020)'].astype(int)
corona_data['Active'] = corona_data['ConfirmedCases'] - corona_data['Fatalities'] 

 

group_data = corona_data.groupby(["Country/Region"])["Fatalities", "ConfirmedCases"].sum().reset_index()

group_data = group_data.sort_values(by='Fatalities', ascending=False)

group_data = group_data[group_data['Fatalities']>100]

plt.figure(figsize=(15, 5))

plt.plot(group_data['Country/Region'], group_data['Fatalities'],color='red')

plt.plot(group_data['Country/Region'], group_data['ConfirmedCases'],color='green')



 

plt.title('Total Deaths(>100), Confirmed Cases by Country')

plt.show()
import pandas as pd

import plotly.express as px

import plotly.io as pio

pio.templates.default = "plotly_dark"

 

grouped = corona_data.groupby('Date')['Date', 'ConfirmedCases', 'Fatalities'].sum().reset_index()

fig = px.line(grouped, x="Date", y="ConfirmedCases",

             title="Worldwide Confirmed Novel Coronavirus(COVID-19) Cases Over Date")

fig.show()
import pandas as pd

import plotly.express as px

import plotly.io as pio

pio.templates.default = "plotly_dark"

 

grouped = corona_data.groupby('Country/Region')['Country/Region', 'Fatalities'].sum().reset_index()

fig = px.line(grouped, x="Country/Region", y="Fatalities",

             title="Worldwide fatalities Novel Coronavirus(COVID-19) Cases Over country")

fig.show()
corona_data.corr()['ConfirmedCases']
#Attributes showing high correlation with dependent variables are not included

X_train=corona_data[['Lat','Long','day','month','Population (2020)','Land Area','Med Age']]
y_train=corona_data[['ConfirmedCases','Fatalities']]
sns.heatmap(X_train.isnull(),yticklabels=False,cbar=False,cmap='viridis')
test_data = test.merge(population, left_on='Country/Region', right_on='Country (or dependency)', how='left')

test_data.shape
test_data['day']=pd.DatetimeIndex(test_data['Date']).day

test_data['year'] = pd.DatetimeIndex(test_data['Date']).year

test_data['month'] = pd.DatetimeIndex(test_data['Date']).month

test_data.head()

test_data.drop('Province/State',axis=1,inplace=True)
X_test=test_data[['Lat','Long','day','month','Population (2020)','Land Area','Med Age']]
X_test[['Population (2020)', 'Land Area', 'Med Age']] = X_test[['Population (2020)', 'Land Area', 'Med Age']].fillna(0)
X_test.info()
# Fitting Polynomial Regression to the dataset

# Fitting Linear Regression to the dataset

from sklearn.linear_model import LinearRegression

from sklearn.preprocessing import PolynomialFeatures

from sklearn.multioutput import MultiOutputRegressor



polynomial_regressor = PolynomialFeatures(degree = 4) #try 2,3 and 4



X_polynomial = polynomial_regressor.fit_transform(X_train)

linear_regressor_2 = LinearRegression()

#for multi-output

regr_multirf = MultiOutputRegressor(linear_regressor_2)

regr_multirf.fit(X_polynomial, y_train)
y_multirf = regr_multirf.predict(polynomial_regressor.fit_transform(X_test))

y_pred = np.round(y_multirf, 1)

y_multirf.shape
y_pred = y_pred.astype(int)
submission = pd.DataFrame(data = np.zeros((y_pred.shape[0],3)), columns = ['ForecastId', 'ConfirmedCases', 'Fatalities'])

submission.shape

y_pred1 = pd.DataFrame(y_pred)
for i in range(0, len(submission)):

    submission.loc[i,'ForecastId'] = i + 1

    submission.loc[i,'ConfirmedCases'] = y_pred1.iloc[i, 0]

    submission.loc[i,'Fatalities'] = y_pred1.iloc[i, 1]
submission['ForecastId'] = submission['ForecastId'].astype(int)

submission['ConfirmedCases'] = submission['ConfirmedCases'].astype(int)

submission['Fatalities'] = submission['Fatalities'].astype(int)
submission
submission.to_csv('submission.csv', index = False)
submission.head()