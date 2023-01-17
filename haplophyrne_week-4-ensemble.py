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

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import datetime as dt
import xgboost as xgb
from xgboost import XGBRegressor



#Reading the cumulative cases dataset
covid_cases = pd.read_csv('../input/novel-corona-virus-2019-dataset/covid_19_data.csv')
training_data = pd.read_csv("../input/covid19-global-forecasting-week-4/train.csv")
testing_data = pd.read_csv("../input/covid19-global-forecasting-week-4/test.csv")
training_data['Province_State'].fillna("",inplace = True)
testing_data['Province_State'].fillna("",inplace = True)
country_list = covid_cases['Country/Region'].unique()

country_grouped_covid = covid_cases[0:1]

for country in country_list:
    test_data = covid_cases['Country/Region'] == country   
    test_data = covid_cases[test_data]
    country_grouped_covid = pd.concat([country_grouped_covid, test_data], axis=0)
    
country_grouped_covid.reset_index(drop=True)
country_grouped_covid.head()

#Dropping of the column Last Update
country_grouped_covid.drop('Last Update', axis=1, inplace=True)

#Replacing NaN Values in Province/State with a string "Not Reported"
country_grouped_covid['Province/State'].replace(np.nan, "Not Reported", inplace=True)
latest_data = country_grouped_covid['ObservationDate'] == '04/10/2020'
country_data = country_grouped_covid[latest_data]

#The total number of reported Countries
country_list = country_data['Country/Region'].unique()
print("The total number of countries with COVID-19 Confirmed cases = {}".format(country_list.size))
unique_dates = country_grouped_covid['ObservationDate'].unique()
confirmed_cases = []
recovered = []
deaths = []

for date in unique_dates:
    date_wise = country_grouped_covid['ObservationDate'] == date  
    test_data = country_grouped_covid[date_wise]
    
    confirmed_cases.append(test_data['Confirmed'].sum())
    deaths.append(test_data['Deaths'].sum())
    recovered.append(test_data['Recovered'].sum())
    
#Converting the lists to a pandas dataframe.

country_dataset = {'Date' : unique_dates, 'Confirmed' : confirmed_cases, 'Recovered' : recovered, 'Deaths' : deaths}
country_dataset = pd.DataFrame(country_dataset)

#Plotting the Graph of Cases vs Deaths Globally.

fig = go.Figure()
fig.add_trace(go.Bar(x=country_dataset['Date'], y=country_dataset['Confirmed'], name='Confirmed Cases of COVID-19', marker_color='rgb(55, 83, 109)'))
fig.add_trace(go.Bar(x=country_dataset['Date'],y=country_dataset['Deaths'],name='Total Deaths because of COVID-19',marker_color='rgb(26, 118, 255)'))

fig.update_layout(title='Confirmed Cases and Deaths from COVID-19',xaxis_tickfont_size=14,
                  yaxis=dict(title='Reported Numbers',titlefont_size=16,tickfont_size=14,),
    legend=dict(x=0,y=1.0,bgcolor='rgba(255, 255, 255, 0)',bordercolor='rgba(255, 255, 255, 0)'),barmode='group',bargap=0.15, bargroupgap=0.1)
fig.show()


fig = go.Figure()
fig.add_trace(go.Bar(x=country_dataset['Date'], y=country_dataset['Confirmed'], name='Confirmed Cases of COVID-19', marker_color='rgb(55, 83, 109)'))
fig.add_trace(go.Bar(x=country_dataset['Date'],y=country_dataset['Recovered'],name='Total Recoveries because of COVID-19',marker_color='rgb(26, 118, 255)'))

fig.update_layout(title='Confirmed Cases and Recoveries from COVID-19',xaxis_tickfont_size=14,
                  yaxis=dict(title='Reported Numbers',titlefont_size=16,tickfont_size=14,),
    legend=dict(x=0,y=1.0,bgcolor='rgba(255, 255, 255, 0)',bordercolor='rgba(255, 255, 255, 0)'),
    barmode='group',bargap=0.15, bargroupgap=0.1)
fig.show()
folder_name = '../input/covcsd-covid19-countries-statistical-dataset'
file_type = 'csv'
seperator =','
dataframe = pd.concat([pd.read_csv(f, sep=seperator) for f in glob.glob(folder_name + "/*."+file_type)],ignore_index=True,sort=False)

#Selecting the columns that are required as is essential for the data-wrangling task

covid_data = dataframe[['Date', 'State', 'Country', 'Cumulative_cases', 'Cumulative_death',
       'Daily_cases', 'Daily_death', 'Latitude', 'Longitude', 'Temperature',
       'Min_temperature', 'Max_temperature', 'Wind_speed', 'Precipitation',
       'Fog_Presence', 'Population', 'Population Density/km', 'Median_Age',
       'Sex_Ratio', 'Age%_65+', 'Hospital Beds/1000', 'Available Beds/1000',
       'Confirmed Cases/1000', 'Lung Patients (F)', 'Lung Patients (M)',
       'Life Expectancy (M)', 'Life Expectancy (F)', 'Total_tests_conducted',
       'Out_Travels (mill.)', 'In_travels(mill.)', 'Domestic_Travels (mill.)']]

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

#Filtering of the dataset to view the latest contents (as of 30-03-2020)
latest_data = covid_data['Date'] == '30-03-2020'
country_data_detailed = covid_data[latest_data]

#Dropping off unecssary columns from the country_data_detailed dataset
country_data_detailed.drop(['Daily_cases','Daily_death','Latitude','Longitude'],axis=1,inplace=True)
country_data_detailed.replace('Not Reported',np.nan,inplace=True)
country_data_detailed.replace('N/A',np.nan,inplace=True)


country_data_detailed['Lung Patients (F)'].replace('Not reported',np.nan,inplace=True)
country_data_detailed['Lung Patients (F)'] = country_data_detailed['Lung Patients (F)'].astype("float")

#Getting the dataset to check the correlation 
corr_data = country_data_detailed.drop(['Date','State','Country','Min_temperature','Max_temperature','Out_Travels (mill.)',
                                        'In_travels(mill.)','Domestic_Travels (mill.)','Total_tests_conducted','Age%_65+'], axis=1)

#Converting the dataset to the correlation function
corr = corr_data.corr()

def heatmap(x, y, size,color):
    fig, ax = plt.subplots(figsize=(20,3))
    
    # Mapping from column names to integer coordinates
    x_labels = corr_data.columns
    y_labels = ['Cumulative_cases', 'Cumulative_death']
    x_to_num = {p[1]:p[0] for p in enumerate(x_labels)} 
    y_to_num = {p[1]:p[0] for p in enumerate(y_labels)} 
    
    n_colors = 256 # Use 256 colors for the diverging color palette
    palette = sns.cubehelix_palette(n_colors) # Create the palette
    color_min, color_max = [-1, 1] # Range of values that will be mapped to the palette, i.e. min and max possible correlation

    def value_to_color(val):
        val_position = float((val - color_min)) / (color_max - color_min) # position of value in the input range, relative to the length of the input range
        ind = int(val_position * (n_colors - 1)) # target index in the color palette
        return palette[ind]

    
    ax.scatter(
    x=x.map(x_to_num),
    y=y.map(y_to_num),
    s=size * 1000,
    c=color.apply(value_to_color), # Vector of square color values, mapped to color palette
    marker='s'
)
    
    # Show column labels on the axes
    ax.set_xticks([x_to_num[v] for v in x_labels])
    ax.set_xticklabels(x_labels, rotation=30, horizontalalignment='right')
    ax.set_yticks([y_to_num[v] for v in y_labels])
    ax.set_yticklabels(y_labels)
    
    
    ax.set_xticks([t + 0.5 for t in ax.get_xticks()], minor=True)
    ax.set_yticks([t + 0.5 for t in ax.get_yticks()], minor=True)
    
    ax.set_xlim([-0.5, max([v for v in x_to_num.values()]) + 0.5]) 
    ax.set_ylim([-0.5, max([v for v in y_to_num.values()]) + 0.5])
    
corr = pd.melt(corr.reset_index(), id_vars='index') 
corr.columns = ['x', 'y', 'value']
heatmap(x=corr['x'],y=corr['y'],size=corr['value'].abs(),color=corr['value'])


#Reading the temperature data file
temperature_data = pd.read_csv('../input/covcsd-covid19-countries-statistical-dataset/temperature_data.csv')

#Viewing the dataset
temperature_data.head()

unique_temp = temperature_data['Temperature'].unique()
confirmed_cases = []
deaths = []

for temp in unique_temp:
    temp_wise = temperature_data['Temperature'] == temp
    test_data = temperature_data[temp_wise]
    
    confirmed_cases.append(test_data['Daily_cases'].sum())
    deaths.append(test_data['Daily_death'].sum())
    
#Converting the lists to a pandas dataframe.

temperature_dataset = {'Temperature' : unique_temp, 'Confirmed' : confirmed_cases, 'Deaths' : deaths}
temperature_dataset = pd.DataFrame(temperature_dataset)



#Plotting a scatter plot for cases vs. Temperature

fig = make_subplots(specs=[[{"secondary_y": True}]])

fig.add_trace(go.Scattergl(x = temperature_dataset['Temperature'],y = temperature_dataset['Confirmed'], mode='markers',
                                  marker=dict(color=np.random.randn(10000),colorscale='Viridis',line_width=1)),secondary_y=False)

fig.add_trace(go.Box(x=temperature_dataset['Temperature']),secondary_y=True)

fig.update_layout(title='Daily Confirmed Cases (COVID-19) vs. Temperature (Celcius) : Global Figures - January 22 - March 30 2020',
                  yaxis=dict(title='Reported Numbers'),xaxis=dict(title='Temperature in Celcius'))

fig.update_yaxes(title_text="BoxPlot Range ", secondary_y=True)
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

rf = XGBRegressor(n_estimators = 1500 , max_depth = 10, learning_rate=0.1)
rf.fit(x_train,cases)
cases_pred = rf.predict(x_test)

rf = XGBRegressor(n_estimators = 1500 , max_depth = 10, learning_rate=0.1)
rf.fit(x_train,fatalities)
fatalities_pred = rf.predict(x_test)

#Rouding off the prediction values and converting negatives to zero
cases_pred = np.around(cases_pred)
fatalities_pred = np.around(fatalities_pred)

cases_pred[cases_pred < 0] = 0
fatalities_pred[fatalities_pred < 0] = 0

submission_dataset = pd.read_csv("../input/covid19-global-forecasting-week-4/submission.csv")

#Adding results to the dataset
submission_dataset['ConfirmedCases'] = cases_pred
submission_dataset['Fatalities'] = fatalities_pred

submission_dataset.head()
submission_dataset.to_csv('submission.csv',index=False)
submission_dataset
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit

%matplotlib inline
%config InlineBackend.figure_format = 'retina'
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import log_loss
from sklearn.preprocessing import OneHotEncoder

import xgboost as xgb
import tensorflow as tf

from tensorflow.keras.optimizers import Nadam
from sklearn.metrics import mean_squared_error
import tensorflow.keras.layers as KL
from datetime import timedelta
import numpy as np
import pandas as pd


import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression, Ridge

import datetime
import gc
from tqdm import tqdm

def get_cpmp_sub(save_oof=False, save_public_test=False):
    train = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/train.csv')
    train['Province_State'].fillna('', inplace=True)
    train['Date'] = pd.to_datetime(train['Date'])
    train['day'] = train.Date.dt.dayofyear
    #train = train[train.day <= 85]
    train['geo'] = ['_'.join(x) for x in zip(train['Country_Region'], train['Province_State'])]
    train

    test = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/test.csv')
    test['Province_State'].fillna('', inplace=True)
    test['Date'] = pd.to_datetime(test['Date'])
    test['day'] = test.Date.dt.dayofyear
    test['geo'] = ['_'.join(x) for x in zip(test['Country_Region'], test['Province_State'])]
    test

    day_min = train['day'].min()
    train['day'] -= day_min
    test['day'] -= day_min

    min_test_val_day = test.day.min()
    max_test_val_day = train.day.max()
    max_test_day = test.day.max()
    num_days = max_test_day + 1

    min_test_val_day, max_test_val_day, num_days

    train['ForecastId'] = -1
    test['Id'] = -1
    test['ConfirmedCases'] = 0
    test['Fatalities'] = 0

    debug = False

    data = pd.concat([train,
                      test[test.day > max_test_val_day][train.columns]
                     ]).reset_index(drop=True)
    if debug:
        data = data[data['geo'] >= 'France_'].reset_index(drop=True)
    #del train, test
    gc.collect()

    dates = data[data['geo'] == 'France_'].Date.values

    if 0:
        gr = data.groupby('geo')
        data['ConfirmedCases'] = gr.ConfirmedCases.transform('cummax')
        data['Fatalities'] = gr.Fatalities.transform('cummax')

    geo_data = data.pivot(index='geo', columns='day', values='ForecastId')
    num_geo = geo_data.shape[0]
    geo_data

    geo_id = {}
    for i,g in enumerate(geo_data.index):
        geo_id[g] = i


    ConfirmedCases = data.pivot(index='geo', columns='day', values='ConfirmedCases')
    Fatalities = data.pivot(index='geo', columns='day', values='Fatalities')

    if debug:
        cases = ConfirmedCases.values
        deaths = Fatalities.values
    else:
        cases = np.log1p(ConfirmedCases.values)
        deaths = np.log1p(Fatalities.values)


    def get_dataset(start_pred, num_train, lag_period):
        days = np.arange( start_pred - num_train + 1, start_pred + 1)
        lag_cases = np.vstack([cases[:, d - lag_period : d] for d in days])
        lag_deaths = np.vstack([deaths[:, d - lag_period : d] for d in days])
        target_cases = np.vstack([cases[:, d : d + 1] for d in days])
        target_deaths = np.vstack([deaths[:, d : d + 1] for d in days])
        geo_ids = np.vstack([geo_ids_base for d in days])
        country_ids = np.vstack([country_ids_base for d in days])
        return lag_cases, lag_deaths, target_cases, target_deaths, geo_ids, country_ids, days

    def update_valid_dataset(data, pred_death, pred_case):
        lag_cases, lag_deaths, target_cases, target_deaths, geo_ids, country_ids, days = data
        day = days[-1] + 1
        new_lag_cases = np.hstack([lag_cases[:, 1:], pred_case])
        new_lag_deaths = np.hstack([lag_deaths[:, 1:], pred_death]) 
        new_target_cases = cases[:, day:day+1]
        new_target_deaths = deaths[:, day:day+1] 
        new_geo_ids = geo_ids  
        new_country_ids = country_ids  
        new_days = 1 + days
        return new_lag_cases, new_lag_deaths, new_target_cases, new_target_deaths, new_geo_ids, new_country_ids, new_days

    def fit_eval(lr_death, lr_case, data, start_lag_death, end_lag_death, num_lag_case, fit, score):
        lag_cases, lag_deaths, target_cases, target_deaths, geo_ids, country_ids, days = data

        X_death = np.hstack([lag_cases[:, -start_lag_death:-end_lag_death], country_ids])
        X_death = np.hstack([lag_deaths[:, -num_lag_case:], country_ids])
        X_death = np.hstack([lag_cases[:, -start_lag_death:-end_lag_death], lag_deaths[:, -num_lag_case:], country_ids])
        y_death = target_deaths
        y_death_prev = lag_deaths[:, -1:]
        if fit:
            if 0:
                keep = (y_death > 0).ravel()
                X_death = X_death[keep]
                y_death = y_death[keep]
                y_death_prev = y_death_prev[keep]
            lr_death.fit(X_death, y_death)
        y_pred_death = lr_death.predict(X_death)
        y_pred_death = np.maximum(y_pred_death, y_death_prev)

        X_case = np.hstack([lag_cases[:, -num_lag_case:], geo_ids])
        X_case = lag_cases[:, -num_lag_case:]
        y_case = target_cases
        y_case_prev = lag_cases[:, -1:]
        if fit:
            lr_case.fit(X_case, y_case)
        y_pred_case = lr_case.predict(X_case)
        y_pred_case = np.maximum(y_pred_case, y_case_prev)

        if score:
            death_score = val_score(y_death, y_pred_death)
            case_score = val_score(y_case, y_pred_case)
        else:
            death_score = 0
            case_score = 0

        return death_score, case_score, y_pred_death, y_pred_case

    def train_model(train, valid, start_lag_death, end_lag_death, num_lag_case, num_val, score=True):
        alpha = 2
        lr_death = Ridge(alpha=alpha, fit_intercept=False)
        lr_case = Ridge(alpha=alpha, fit_intercept=True)

        (train_death_score, train_case_score, train_pred_death, train_pred_case,
        ) = fit_eval(lr_death, lr_case, train, start_lag_death, end_lag_death, num_lag_case, fit=True, score=score)

        death_scores = []
        case_scores = []

        death_pred = []
        case_pred = []

        for i in range(num_val):

            (valid_death_score, valid_case_score, valid_pred_death, valid_pred_case,
            ) = fit_eval(lr_death, lr_case, valid, start_lag_death, end_lag_death, num_lag_case, fit=False, score=score)

            death_scores.append(valid_death_score)
            case_scores.append(valid_case_score)
            death_pred.append(valid_pred_death)
            case_pred.append(valid_pred_case)

            if 0:
                print('val death: %0.3f' %  valid_death_score,
                      'val case: %0.3f' %  valid_case_score,
                      'val : %0.3f' %  np.mean([valid_death_score, valid_case_score]),
                      flush=True)
            valid = update_valid_dataset(valid, valid_pred_death, valid_pred_case)

        if score:
            death_scores = np.sqrt(np.mean([s**2 for s in death_scores]))
            case_scores = np.sqrt(np.mean([s**2 for s in case_scores]))
            if 0:
                print('train death: %0.3f' %  train_death_score,
                      'train case: %0.3f' %  train_case_score,
                      'val death: %0.3f' %  death_scores,
                      'val case: %0.3f' %  case_scores,
                      'val : %0.3f' % ( (death_scores + case_scores) / 2),
                      flush=True)
            else:
                print('%0.4f' %  case_scores,
                      ', %0.4f' %  death_scores,
                      '= %0.4f' % ( (death_scores + case_scores) / 2),
                      flush=True)
        death_pred = np.hstack(death_pred)
        case_pred = np.hstack(case_pred)
        return death_scores, case_scores, death_pred, case_pred

    countries = [g.split('_')[0] for g in geo_data.index]
    countries = pd.factorize(countries)[0]

    country_ids_base = countries.reshape((-1, 1))
    ohe = OneHotEncoder(sparse=False)
    country_ids_base = 0.2 * ohe.fit_transform(country_ids_base)
    country_ids_base.shape

    geo_ids_base = np.arange(num_geo).reshape((-1, 1))
    ohe = OneHotEncoder(sparse=False)
    geo_ids_base = 0.1 * ohe.fit_transform(geo_ids_base)
    geo_ids_base.shape

    def val_score(true, pred):
        pred = np.log1p(np.round(np.expm1(pred) - 0.2))
        return np.sqrt(mean_squared_error(true.ravel(), pred.ravel()))

    def val_score(true, pred):
        return np.sqrt(mean_squared_error(true.ravel(), pred.ravel()))



    start_lag_death, end_lag_death = 14, 6,
    num_train = 6
    num_lag_case = 14
    lag_period = max(start_lag_death, num_lag_case)

    def get_oof(start_val_delta=0):   
        start_val = min_test_val_day + start_val_delta
        last_train = start_val - 1
        num_val = max_test_val_day - start_val + 1
        print(dates[start_val], start_val, num_val)
        train_data = get_dataset(last_train, num_train, lag_period)
        valid_data = get_dataset(start_val, 1, lag_period)
        _, _, val_death_preds, val_case_preds = train_model(train_data, valid_data, 
                                                            start_lag_death, end_lag_death, num_lag_case, num_val)

        pred_deaths = Fatalities.iloc[:, start_val:start_val+num_val].copy()
        pred_deaths.iloc[:, :] = np.expm1(val_death_preds)
        pred_deaths = pred_deaths.stack().reset_index()
        pred_deaths.columns = ['geo', 'day', 'Fatalities']
        pred_deaths

        pred_cases = ConfirmedCases.iloc[:, start_val:start_val+num_val].copy()
        pred_cases.iloc[:, :] = np.expm1(val_case_preds)
        pred_cases = pred_cases.stack().reset_index()
        pred_cases.columns = ['geo', 'day', 'ConfirmedCases']
        pred_cases

        sub = train[['Date', 'Id', 'geo', 'day']]
        sub = sub.merge(pred_cases, how='left', on=['geo', 'day'])
        sub = sub.merge(pred_deaths, how='left', on=['geo', 'day'])
        #sub = sub.fillna(0)
        sub = sub[sub.day >= start_val]
        sub = sub[['Id', 'ConfirmedCases', 'Fatalities']].copy()
        return sub


    if save_oof:
        for start_val_delta, date in zip(range(3, -8, -3),
                                  ['2020-03-22', '2020-03-19', '2020-03-16', '2020-03-13']):
            print(date, end=' ')
            oof = get_oof(start_val_delta)
            oof.to_csv('../submissions/cpmp-%s.csv' % date, index=None)

    def get_sub(start_val_delta=0):   
        start_val = min_test_val_day + start_val_delta
        last_train = start_val - 1
        num_val = max_test_val_day - start_val + 1
        print(dates[last_train], start_val, num_val)
        num_lag_case = 14
        train_data = get_dataset(last_train, num_train, lag_period)
        valid_data = get_dataset(start_val, 1, lag_period)
        _, _, val_death_preds, val_case_preds = train_model(train_data, valid_data, 
                                                            start_lag_death, end_lag_death, num_lag_case, num_val)

        pred_deaths = Fatalities.iloc[:, start_val:start_val+num_val].copy()
        pred_deaths.iloc[:, :] = np.expm1(val_death_preds)
        pred_deaths = pred_deaths.stack().reset_index()
        pred_deaths.columns = ['geo', 'day', 'Fatalities']
        pred_deaths

        pred_cases = ConfirmedCases.iloc[:, start_val:start_val+num_val].copy()
        pred_cases.iloc[:, :] = np.expm1(val_case_preds)
        pred_cases = pred_cases.stack().reset_index()
        pred_cases.columns = ['geo', 'day', 'ConfirmedCases']
        pred_cases

        sub = test[['Date', 'ForecastId', 'geo', 'day']]
        sub = sub.merge(pred_cases, how='left', on=['geo', 'day'])
        sub = sub.merge(pred_deaths, how='left', on=['geo', 'day'])
        sub = sub.fillna(0)
        sub = sub[['ForecastId', 'ConfirmedCases', 'Fatalities']]
        return sub
        return sub


    known_test = train[['geo', 'day', 'ConfirmedCases', 'Fatalities']
              ].merge(test[['geo', 'day', 'ForecastId']], how='left', on=['geo', 'day'])
    known_test = known_test[['ForecastId', 'ConfirmedCases', 'Fatalities']][known_test.ForecastId.notnull()].copy()
    known_test

    unknow_test = test[test.day > max_test_val_day]
    unknow_test

    def get_final_sub():   
        start_val = max_test_val_day + 1
        last_train = start_val - 1
        num_val = max_test_day - start_val + 1
        print(dates[last_train], start_val, num_val)
        num_lag_case = num_val + 3
        train_data = get_dataset(last_train, num_train, lag_period)
        valid_data = get_dataset(start_val, 1, lag_period)
        (_, _, val_death_preds, val_case_preds
        ) = train_model(train_data, valid_data, start_lag_death, end_lag_death, num_lag_case, num_val, score=False)

        pred_deaths = Fatalities.iloc[:, start_val:start_val+num_val].copy()
        pred_deaths.iloc[:, :] = np.expm1(val_death_preds)
        pred_deaths = pred_deaths.stack().reset_index()
        pred_deaths.columns = ['geo', 'day', 'Fatalities']
        pred_deaths

        pred_cases = ConfirmedCases.iloc[:, start_val:start_val+num_val].copy()
        pred_cases.iloc[:, :] = np.expm1(val_case_preds)
        pred_cases = pred_cases.stack().reset_index()
        pred_cases.columns = ['geo', 'day', 'ConfirmedCases']
        pred_cases
        print(unknow_test.shape, pred_deaths.shape, pred_cases.shape)

        sub = unknow_test[['Date', 'ForecastId', 'geo', 'day']]
        sub = sub.merge(pred_cases, how='left', on=['geo', 'day'])
        sub = sub.merge(pred_deaths, how='left', on=['geo', 'day'])
        #sub = sub.fillna(0)
        sub = sub[['ForecastId', 'ConfirmedCases', 'Fatalities']]
        sub = pd.concat([known_test, sub])
        return sub

    if save_public_test:
        sub = get_sub()
    else:
        sub = get_final_sub()
    return sub

def get_nn_sub():
    df = pd.read_csv("../input/covid19-global-forecasting-week-4/train.csv")
    sub_df = pd.read_csv("../input/covid19-global-forecasting-week-4/test.csv")

    coo_df = pd.read_csv("../input/week11/train.csv").rename(columns={"Country/Region": "Country_Region"})
    coo_df = coo_df.groupby("Country_Region")[["Lat", "Long"]].mean().reset_index()
    coo_df = coo_df[coo_df["Country_Region"].notnull()]

    loc_group = ["Province_State", "Country_Region"]


    def preprocess(df):
        df["Date"] = df["Date"].astype("datetime64[ms]")
        df["days"] = (df["Date"] - pd.to_datetime("2020-01-01")).dt.days
        df["weekend"] = df["Date"].dt.dayofweek//5

        df = df.merge(coo_df, how="left", on="Country_Region")
        df["Lat"] = (df["Lat"] // 30).astype(np.float32).fillna(0)
        df["Long"] = (df["Long"] // 60).astype(np.float32).fillna(0)

        for col in loc_group:
            df[col].fillna("none", inplace=True)
        return df

    df = preprocess(df)
    sub_df = preprocess(sub_df)

    print(df.shape)

    TARGETS = ["ConfirmedCases", "Fatalities"]

    for col in TARGETS:
        df[col] = np.log1p(df[col])

    NUM_SHIFT = 5

    features = ["Lat", "Long"]

    for s in range(1, NUM_SHIFT+1):
        for col in TARGETS:
            df["prev_{}_{}".format(col, s)] = df.groupby(loc_group)[col].shift(s)
            features.append("prev_{}_{}".format(col, s))

    df = df[df["Date"] >= df["Date"].min() + timedelta(days=NUM_SHIFT)].copy()

    TEST_FIRST = sub_df["Date"].min() # pd.to_datetime("2020-03-13") #
    TEST_DAYS = (df["Date"].max() - TEST_FIRST).days + 1

    dev_df, test_df = df[df["Date"] < TEST_FIRST].copy(), df[df["Date"] >= TEST_FIRST].copy()

    def nn_block(input_layer, size, dropout_rate, activation):
        out_layer = KL.Dense(size, activation=None)(input_layer)
        #out_layer = KL.BatchNormalization()(out_layer)
        out_layer = KL.Activation(activation)(out_layer)
        out_layer = KL.Dropout(dropout_rate)(out_layer)
        return out_layer


    def get_model():
        inp = KL.Input(shape=(len(features),))

        hidden_layer = nn_block(inp, 622, 0.0, "relu")
        gate_layer = nn_block(hidden_layer, 311, 0.0, "hard_sigmoid")
        hidden_layer = nn_block(hidden_layer, 311, 0.0, "relu")
        hidden_layer = KL.multiply([hidden_layer, gate_layer])

        out = KL.Dense(len(TARGETS), activation="linear")(hidden_layer)

        model = tf.keras.models.Model(inputs=[inp], outputs=out)
        return model

    get_model().summary()

    def get_input(df):
        return [df[features]]

    NUM_MODELS = 100


    def train_models(df, save=False):
        models = []
        for i in range(NUM_MODELS):
            model = get_model()
            model.compile(loss="mean_squared_error", optimizer=Nadam(lr=1e-4))
            hist = model.fit(get_input(df), df[TARGETS],
                             batch_size=2250, epochs=1000, verbose=0, shuffle=True)
            if save:
                model.save_weights("model{}.h5".format(i))
            models.append(model)
        return models

    models = train_models(dev_df)


    prev_targets = ['prev_ConfirmedCases_1', 'prev_Fatalities_1']

    def predict_one(df, models):
        pred = np.zeros((df.shape[0], 2))
        for model in models:
            pred += model.predict(get_input(df))/len(models)
        pred = np.maximum(pred, df[prev_targets].values)
        pred[:, 0] = np.log1p(np.expm1(pred[:, 0]) + 0.1)
        pred[:, 1] = np.log1p(np.expm1(pred[:, 1]) + 0.01)
        return np.clip(pred, None, 15)

    print([mean_squared_error(dev_df[TARGETS[i]], predict_one(dev_df, models)[:, i]) for i in range(len(TARGETS))])


    def rmse(y_true, y_pred):
        return np.sqrt(mean_squared_error(y_true, y_pred))

    def evaluate(df):
        error = 0
        for col in TARGETS:
            error += rmse(df[col].values, df["pred_{}".format(col)].values)
        return np.round(error/len(TARGETS), 5)


    def predict(test_df, first_day, num_days, models, val=False):
        temp_df = test_df.loc[test_df["Date"] == first_day].copy()
        y_pred = predict_one(temp_df, models)

        for i, col in enumerate(TARGETS):
            test_df["pred_{}".format(col)] = 0
            test_df.loc[test_df["Date"] == first_day, "pred_{}".format(col)] = y_pred[:, i]

        print(first_day, np.isnan(y_pred).sum(), y_pred.min(), y_pred.max())
        if val:
            print(evaluate(test_df[test_df["Date"] == first_day]))


        y_prevs = [None]*NUM_SHIFT

        for i in range(1, NUM_SHIFT):
            y_prevs[i] = temp_df[['prev_ConfirmedCases_{}'.format(i), 'prev_Fatalities_{}'.format(i)]].values

        for d in range(1, num_days):
            date = first_day + timedelta(days=d)
            print(date, np.isnan(y_pred).sum(), y_pred.min(), y_pred.max())

            temp_df = test_df.loc[test_df["Date"] == date].copy()
            temp_df[prev_targets] = y_pred
            for i in range(2, NUM_SHIFT+1):
                temp_df[['prev_ConfirmedCases_{}'.format(i), 'prev_Fatalities_{}'.format(i)]] = y_prevs[i-1]

            y_pred, y_prevs = predict_one(temp_df, models), [None, y_pred] + y_prevs[1:-1]


            for i, col in enumerate(TARGETS):
                test_df.loc[test_df["Date"] == date, "pred_{}".format(col)] = y_pred[:, i]

            if val:
                print(evaluate(test_df[test_df["Date"] == date]))

        return test_df

    test_df = predict(test_df, TEST_FIRST, TEST_DAYS, models, val=True)
    print(evaluate(test_df))

    for col in TARGETS:
        test_df[col] = np.expm1(test_df[col])
        test_df["pred_{}".format(col)] = np.expm1(test_df["pred_{}".format(col)])

    models = train_models(df, save=True)

    sub_df_public = sub_df[sub_df["Date"] <= df["Date"].max()].copy()
    sub_df_private = sub_df[sub_df["Date"] > df["Date"].max()].copy()

    pred_cols = ["pred_{}".format(col) for col in TARGETS]
    #sub_df_public = sub_df_public.merge(test_df[["Date"] + loc_group + pred_cols].rename(columns={col: col[5:] for col in pred_cols}), 
    #                                    how="left", on=["Date"] + loc_group)
    sub_df_public = sub_df_public.merge(test_df[["Date"] + loc_group + TARGETS], how="left", on=["Date"] + loc_group)

    SUB_FIRST = sub_df_private["Date"].min()
    SUB_DAYS = (sub_df_private["Date"].max() - sub_df_private["Date"].min()).days + 1

    sub_df_private = df.append(sub_df_private, sort=False)

    for s in range(1, NUM_SHIFT+1):
        for col in TARGETS:
            sub_df_private["prev_{}_{}".format(col, s)] = sub_df_private.groupby(loc_group)[col].shift(s)

    sub_df_private = sub_df_private[sub_df_private["Date"] >= SUB_FIRST].copy()

    sub_df_private = predict(sub_df_private, SUB_FIRST, SUB_DAYS, models)

    for col in TARGETS:
        sub_df_private[col] = np.expm1(sub_df_private["pred_{}".format(col)])

    sub_df = sub_df_public.append(sub_df_private, sort=False)
    sub_df["ForecastId"] = sub_df["ForecastId"].astype(np.int16)

    return sub_df[["ForecastId"] + TARGETS]

sub1 = get_cpmp_sub()
sub1['ForecastId'] = sub1['ForecastId'].astype('int')
sub2 = get_nn_sub()

sub1.sort_values("ForecastId", inplace=True)
sub2.sort_values("ForecastId", inplace=True)


from sklearn.metrics import mean_squared_error

TARGETS = ["ConfirmedCases", "Fatalities"]

[np.sqrt(mean_squared_error(np.log1p(sub1[t].values), np.log1p(sub2[t].values))) for t in TARGETS]

sub_df = sub1.copy()
for t in TARGETS:
    sub_df[t] = np.expm1(np.log1p(sub1[t].values)*0.5 + np.log1p(sub2[t].values)*0.5)
import pandas as pd
from pathlib import Path
from pandas_profiling import ProfileReport
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
import datetime
from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score

dataset_path = Path('/kaggle/input/covid19-global-forecasting-week-4')

train = pd.read_csv(dataset_path/'train.csv')
test = pd.read_csv(dataset_path/'test.csv')
submission = pd.read_csv(dataset_path/'submission.csv')

def fill_state(state,country):
    if pd.isna(state) : return country
    return state

train['Province_State'] = train.loc[:, ['Province_State', 'Country_Region']].apply(lambda x : fill_state(x['Province_State'], x['Country_Region']), axis=1)
test['Province_State'] = test.loc[:, ['Province_State', 'Country_Region']].apply(lambda x : fill_state(x['Province_State'], x['Country_Region']), axis=1)

train['Date'] = pd.to_datetime(train['Date'],infer_datetime_format=True)
test['Date'] = pd.to_datetime(test['Date'],infer_datetime_format=True)

train['Day_of_Week'] = train['Date'].dt.dayofweek
test['Day_of_Week'] = test['Date'].dt.dayofweek

train['Month'] = train['Date'].dt.month
test['Month'] = test['Date'].dt.month

train['Day'] = train['Date'].dt.day
test['Day'] = test['Date'].dt.day

train['Day_of_Year'] = train['Date'].dt.dayofyear
test['Day_of_Year'] = test['Date'].dt.dayofyear

train['Week_of_Year'] = train['Date'].dt.weekofyear
test['Week_of_Year'] = test['Date'].dt.weekofyear

train['Quarter'] = train['Date'].dt.quarter  
test['Quarter'] = test['Date'].dt.quarter  

train.drop('Date',1,inplace=True)
test.drop('Date',1,inplace=True)

submission=pd.DataFrame(columns=submission.columns)

l1=LabelEncoder()
l2=LabelEncoder()

l1.fit(train['Country_Region'])
l2.fit(train['Province_State'])

countries=train['Country_Region'].unique()
for country in countries:
    country_df=train[train['Country_Region']==country]
    provinces=country_df['Province_State'].unique()
    for province in provinces:
            train_df=country_df[country_df['Province_State']==province]
            train_df.pop('Id')
            x=train_df[['Province_State','Country_Region','Day_of_Week','Month','Day','Day_of_Year','Week_of_Year','Quarter']]
            x['Country_Region']=l1.transform(x['Country_Region'])
            x['Province_State']=l2.transform(x['Province_State'])
            y1=train_df[['ConfirmedCases']]
            y2=train_df[['Fatalities']]
            model_1=DecisionTreeClassifier()
            model_2=DecisionTreeClassifier()
            model_1.fit(x,y1)
            model_2.fit(x,y2)
            test_df=test.query('Province_State==@province & Country_Region==@country')
            test_id=test_df['ForecastId'].values.tolist()
            test_df.pop('ForecastId')
            test_x=test_df[['Province_State','Country_Region','Day_of_Week','Month','Day','Day_of_Year','Week_of_Year','Quarter']]
            test_x['Country_Region']=l1.transform(test_x['Country_Region'])
            test_x['Province_State']=l2.transform(test_x['Province_State'])
            test_y1=model_1.predict(test_x)
            test_y2=model_2.predict(test_x)
            test_res=pd.DataFrame(columns=submission.columns)
            test_res['ForecastId']=test_id
            test_res['ConfirmedCases']=test_y1
            test_res['Fatalities']=test_y2
            submission=submission.append(test_res)





\
lcc=submission["ConfirmedCases"]
lf=submission["Fatalities"]

buscc = submission_dataset["ConfirmedCases"]
busf = submission_dataset["Fatalities"]
sdfcc = sub_df["ConfirmedCases"]
sdff = sub_df["Fatalities"]
sub_df["ConfirmedCases"] = 0.1 * buscc.values +  0.70 * sdfcc.values + 0.20 *lcc.values
sub_df["Fatalities"] = 0.1 * busf.values  +  0.70 * sdff.values + 0.20  * lf.values
sub_df.to_csv('submission.csv',index=False)