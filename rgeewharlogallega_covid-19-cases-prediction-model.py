# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

plt.style.use('ggplot')

import sklearn
import datetime as dt

# Input data files are available in the "../input/" directory.

# Any results you write to the current directory are saved as output.

FIG_SIZE = (15,3)
# filename = '/kaggle/input/corona-virus-report/covid_19_clean_complete.csv'
file_confirmed = '/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_confirmed.csv'
file_deaths = '/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_deaths.csv'
file_recovered = '/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_recovered.csv'


df_confirmed = pd.read_csv(file_confirmed)
df_deaths = pd.read_csv(file_deaths)
df_recovered = pd.read_csv(file_recovered)

# df['Date'] = pd.to_datetime(df['Date'])

print(df_confirmed.head())
dates = df_confirmed.iloc[:,4:].columns
dates_ = pd.to_datetime(dates)
dates_ = pd.to_datetime(dates_)

countries = df_confirmed['Country/Region'].unique()

latest = str(max(dates_).strftime('%-m/%-d/%y'))


df_new_c = df_confirmed[['Country/Region',latest]].groupby('Country/Region',as_index=False).sum().sort_values([latest])
df_new_c = df_new_c.rename(columns={latest:'Confirmed'})

df_new_d = df_deaths[['Country/Region',latest]].groupby('Country/Region',as_index=False).sum().sort_values([latest])
df_new_d = df_new_d.rename(columns={latest:'Deaths'})

df_new_r = df_recovered[['Country/Region',latest]].groupby('Country/Region',as_index=False).sum().sort_values([latest])
df_new_r = df_new_r.rename(columns={latest:'Recovered'})

df_latest = pd.merge(df_new_c, df_new_d, on='Country/Region')
df_latest = pd.merge(df_latest, df_new_r, on ='Country/Region')

df_latest.sort_values(['Confirmed']).plot(kind='barh', stacked='true',figsize=(10,30), x='Country/Region', color=['r','k','b'])

df_latest_ph = df_latest.loc[df_latest['Country/Region']=='Philippines']
df_latest_ph.plot(kind='bar',x='Country/Region',color=['r','k','b'],rot=0)
df_time_c = df_confirmed.drop(['Lat', 'Long'],axis=1).groupby('Country/Region',as_index=False).sum()
df_time_d = df_deaths.drop(['Lat', 'Long'],axis=1).groupby('Country/Region',as_index=False).sum()
df_time_r = df_recovered.drop(['Lat', 'Long'],axis=1).groupby('Country/Region',as_index=False).sum()

df_time_cT = df_time_c.set_index('Country/Region').transpose().rename(index={'Country/Region':'Dates'})
df_time_dT = df_time_d.set_index('Country/Region').transpose().rename(index={'Country/Region':'Dates'})
df_time_rT = df_time_r.set_index('Country/Region').transpose().rename(index={'Country/Region':'Dates'})

sel_countries = ['Philippines','US','China','Italy','Korea, South','Spain']

# df_time_ph = df_time_cT['Philippines'].to_frame()
# df_time_ph = df_time_ph.merge(df_time_dT['Philippines'], left_index=True, right_index=True)
# df_time_ph = df_time_ph.merge(df_time_rT['Philippines'], left_index=True, right_index=True)
# df_time_ph.columns = ['Confirmed','Deaths','Recovered']
# df_time_ph.plot(kind='line',ax=axes[0],figsize=FIG_SIZE)

df_time_ph = None
df_time_us = None


for i in range(len(sel_countries)):
    country = sel_countries[i]
    df_time_country = df_time_cT[country].to_frame()
    df_time_country = df_time_country.merge(df_time_dT[country], left_index=True, right_index=True)
    df_time_country = df_time_country.merge(df_time_rT[country], left_index=True, right_index=True)
    df_time_country.columns = ['Confirmed','Deaths','Recovered']
    if country=='Philippines':
        df_time_ph = df_time_country
    if country=='US':
        df_time_us = df_time_country
    df_time_country.plot(kind='line',figsize=FIG_SIZE, title='COVID-19 in {}'.format(country))

from sklearn.model_selection import train_test_split as tts

covid_days = np.array([i for i in range(len(dates_))]).reshape(-1, 1)
values = df_time_ph['Confirmed'].values.reshape(-1,1)

# covid_days = np.append(covid_days, 74)
# values = np.append(values, 3246)

X = covid_days
y = values

X_train, X_test, y_train, y_test = tts(X,y, test_size=0.25, shuffle=False)

plt.scatter(X,y)
def poly_feature_transform(X, order=1):
    """
    Transforms the input data to match the specified polynomial order.

    Inputs:
    - X: A numpy array of shape (N, D) consisting
         of N samples each of dimension D.
    - order: Determines the order of the polynomial of the hypothesis function.

    Returns:
    - f_transform: A numpy array of shape (N, (D * order) + 1) representing the transformed
        features following the specified order.
    """
    if(order==1):
        ones = np.ones(len(X))
        np.expand_dims(ones,1)
        f_transform = np.append(X,ones,axis=1)
        print(f_transform)
        return f_transform
    else:
        f_transform = X
        for i in range(2,order+1):
            addtl_order = np.power(X,i)
            np.expand_dims(addtl_order,axis=1)
            f_transform= np.append(f_transform,addtl_order,axis=1)
        ones = np.ones(len(X))
        ones = np.expand_dims(ones,1)
        f_transform = np.append(f_transform,ones,axis=1)
    
        return f_transform
    return None
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import BayesianRidge
from sklearn.linear_model import SGDRegressor
from sklearn.svm import SVR

from sklearn.metrics import mean_squared_error as MSE

linreg = LinearRegression()
ridge = Ridge(max_iter=3000000, alpha=0.015, tol=0.00001,normalize=True)
lasso = Lasso(max_iter=3000000, alpha=0.001, tol=0.01,normalize=True)
en = ElasticNet()
baye = BayesianRidge()
svr = SVR(kernel='poly',C=0.1,max_iter=3000000)
sgd = SGDRegressor(loss='huber')

# models = [linreg,ridge,lasso,en,baye,sgd,svr]
models = [lasso,ridge]

order = 10

for model in models:
    predictions = None
    if(model == ridge or model ==lasso or model == linreg):
        new_x_train = poly_feature_transform(X_train,order)
        model.fit(new_x_train,y_train)
        new_x_test = poly_feature_transform(X_test,order)
        predictions = model.predict(new_x_test)
    else:
        model.fit(X_train,y_train)
        predictions = model.predict(X_test)
    print("Mean Squared Error of {}: {}".format(type(model),MSE(y_test,predictions)))
    plt.scatter(X_test,y_test,color='k')
    plt.plot(X_test,predictions)
# def predict(day):
    
covid_start_date = min(dates_)

print("Type in Q to quit.\n")
while(True):
    print("Date (MM/DD/YYYY):")
    input_date = input()
    if(input_date == "Q" or input_date=="q"):
        break;
    else:
        d = pd.to_datetime(input_date) 
        days = (d-covid_start_date).days
        days = np.expand_dims([days],axis=1)
        days = poly_feature_transform(days,order)
        for model in models:
            pred = model.predict(days)
            model_name = type(model).__name__
            print("On date", input_date , ", the estimated number of cases is", np.squeeze(pred), "predicted by Jeongyeon (", model_name, ").\n" )