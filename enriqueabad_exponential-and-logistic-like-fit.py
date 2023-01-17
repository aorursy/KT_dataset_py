# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from matplotlib import pyplot as plt

import matplotlib.dates as mdates

from datetime import timedelta

from scipy.optimize import curve_fit

from scipy.stats import linregress

from scipy.special import erf

from sklearn.metrics import mean_squared_error

import warnings



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train_data = pd.read_csv('../input/covid19-global-forecasting-week-1/train.csv')

test_data = pd.read_csv('../input/covid19-global-forecasting-week-1/test.csv')



# Change column names: '/' character may cause problems



train_data = train_data.rename(columns={ 'Province/State' : 'State', 'Country/Region' : 'Country',

                                         'Date' : 'DateAsString' })



test_data = test_data.rename(columns={ 'Province/State' : 'State', 'Country/Region' : 'Country',

                                         'Date' : 'DateAsString' })



# Put dates as datetime64 datatype



train_data['Date'] = pd.to_datetime(train_data['DateAsString'], format='%Y-%m-%d')

test_data['Date'] = pd.to_datetime(test_data['DateAsString'], format='%Y-%m-%d')



# If there are no states, there is only one state, called 'All'



train_data['State'] = train_data['State'].fillna('All')

test_data['State'] = test_data['State'].fillna('All')



# Take out aposthrophes in countries with aposthropes (it messes with the string definitions!)



train_data = train_data.replace("Cote d'Ivoire","Cote d Ivoire")

test_data = test_data.replace("Cote d'Ivoire","Cote d Ivoire")





# Add logaritmic values, because it is often the best metric here



train_data[['LogConfirmed']] = train_data[['ConfirmedCases']].apply(np.log)

train_data[['LogFatalities']] = train_data[['Fatalities']].apply(np.log)



test_data['ConfirmedPred'] = np.zeros(test_data.shape[0])

test_data['FatalitiesPred'] = np.zeros(test_data.shape[0])
last_day = np.datetime64('2020-04-23')

first_prediction = np.datetime64('2020-03-12')
train_data_korea = train_data.query("Country == 'Korea, South'")



train_data_korea_100 = train_data_korea.query("ConfirmedCases > 100")



plt.plot(train_data_korea_100['LogConfirmed'])

plt.plot(train_data_korea_100['LogFatalities'])
def logistic_to_fit(x,k,L):

    return L/(1 + np.exp(-k*x))



def error_to_fit(x,k,L):

    return L*(1 + erf(k*x))



def log_normal_to_fit(x,k,L,x_0):

    return L*(1+ erf(np.log(x)-x_0)/k)



x_tofit = np.arange(train_data_korea_100.shape[0])

y_tofit = train_data_korea_100['LogConfirmed'].to_numpy()



model_to_fit = [ logistic_to_fit, error_to_fit, log_normal_to_fit]

popt_confirmed = []

pcov_confirmed = []



for i_model in model_to_fit:

    i_popt, i_pcov = curve_fit(i_model, x_tofit, y_tofit)

    popt_confirmed.append(i_popt)

    pcov_confirmed.append(i_pcov)

    mse = mean_squared_error(y_tofit,i_model(x_tofit,*i_popt))

    print('For model {0} the MSE is {1}'.format(str(i_model).split()[1],mse))
def do_fitting(state_data):

    x_tofit = np.arange(state_data.shape[0])

    y_tofit = state_data['LogConfirmed'].to_numpy()

    offset_array = state_data['LogConfirmed'].to_numpy() - state_data['LogFatalities'].to_numpy()

    offset = offset_array.mean()

    not_fit = False

    warnings.filterwarnings('error')

    try:

        popt, pcov = curve_fit(logistic_to_fit, x_tofit, y_tofit, p0=[0.23, 9])

    except Warning:

        not_fit = True

    slope, intercept, rvalue, pvalue, stderr = linregress(x_tofit, y_tofit)

    warnings.resetwarnings()



    if not_fit:

        def linear_fitted(x):

            y = intercept + slope * x

            return y

        def logistics_fitted(x):

            y = intercept + slope * x

            return y



        mse_logistic = 0.5

        mse_linear = 0.5

        popt = (1e+10,1e+10)



    else:

        def logistics_fitted(x):

            return logistic_to_fit(x,*popt)

        def linear_fitted(x):

            y = intercept + slope * x

            return y



        mse_logistic = mean_squared_error(y_tofit, logistics_fitted(x_tofit))

        mse_linear = mean_squared_error(y_tofit,linear_fitted(x_tofit))



    weight_logistic2 = mse_linear*mse_linear / (mse_linear*mse_linear+mse_logistic*mse_logistic)

    weight_linear2 = mse_logistic*mse_logistic / (mse_linear * mse_linear + mse_logistic * mse_logistic)



    def weighted_fitted(x):

        y = weight_logistic2 * logistics_fitted(x) + weight_linear2 * linear_fitted(x)

        return y



    return (weighted_fitted,logistics_fitted,linear_fitted,weight_logistic2,slope,np.exp(popt[1]),offset)

def predictions_100plus(key,dictionary,plot_it=True):

    country_values = dictionary[key]

    country, state = key.split('; ')

    texto = ''

    weighted_fitted = country_values[0]

    logistic_fitted = country_values[1]

    linear_fitted = country_values[2]

    weight_logistic2 = country_values[3]

    slope = country_values[4]

    stabilization = round(country_values[5])

    offset = country_values[6]

    first_day = country_values[7]

    first_day_100 = country_values[8]

    if weight_logistic2 > 0.5:

        texto = texto + 'Stabilyzing phase. Odds {0:4.2f} to 1.\n'.format(weight_logistic2/(1-weight_logistic2))

    else:

        texto = texto + 'Exponential phase. Odds {0:4.2f} to 1\n'.format((1-weight_logistic2)/weight_logistic2)

    t_double = np.log(2) / slope

    texto = texto + 'Duplication every {0:5.2f} days\n'.format(t_double)

    texto = texto + 'The state may stabilize with {0} cases\n'.format(stabilization)



    difference = first_day_100 - first_day

    first_day_int = difference.astype(int)

    difference = last_day - first_day_100

    last_day_int = difference.astype(int)

    x_tofit = np.arange(-first_day_int, last_day_int)

    x_date = np.arange(first_day, last_day)

    y_exponential = np.exp(linear_fitted(x_tofit))

    y_logistic = np.exp(logistic_fitted(x_tofit))

    y_weighted = np.exp(weighted_fitted(x_tofit))

    y_fatalities = np.exp(weighted_fitted(x_tofit) - offset)



    if plot_it:

        fig = plt.figure()

        plt.plot(x_date, y_exponential, label='Exponential fit')

        # This is because the annoying way of displaying dates by matplotlib

        plt.xticks(np.arange(np.datetime64('2020-01-30'), np.datetime64('2020-04-15'),timedelta(days=15)),

                   labels=np.datetime_as_string(np.arange(np.datetime64('2020-01-30'), np.datetime64('2020-04-15'),timedelta(days=15)),unit='D'))

        plt.plot(x_date, y_logistic, label='Logistic fit')

        plt.plot(x_date, y_weighted, label='Weighted fit')

        plt.plot(x_date, y_fatalities, label='Fatalities fit')

        plt.title('{0} ({1})'.format(country, state))

        plt.legend()

        plt.text(first_day, 4.9, texto)

        fig.autofmt_xdate()

    else:

        for i_x, i_date in enumerate(x_date):

            if i_date >= first_prediction:

                query_text = "Country == '{0}' & State == '{1}' & Date == '{2}'".format(country,state,i_date)

                index = test_data.query(query_text)['ForecastId'].to_numpy()[0]

                y_weighted = np.exp(weighted_fitted(x_tofit[i_x]))

                y_fatalities = np.exp(weighted_fitted(x_tofit[i_x]) - offset)

                test_data.at[index,'ConfirmedPred'] = y_weighted

                test_data.at[index,'FatalitiesPred'] = y_fatalities



def predictions_100minus(country,state,data_this_state):

    #print(data_this_state)

    avg_slope = 0.19266443693901186

    avg_offset = 4.577772921762378

    avg_growth_rate = np.exp(avg_slope)

    avg_fatality_rate = np.exp(-avg_offset)

    first_date = data_this_state.head(1)['Date'].to_numpy()[0]

    last_date = data_this_state.tail(1)['Date'].to_numpy()[0]

    this_date = first_date

    while this_date <= last_date:

        query_text = "Country == '{0}' & State == '{1}' & Date == '{2}'".format(country, state, this_date)

        try:

            index_test = test_data.query(query_text)['ForecastId'].index[0]

            index_train = train_data.query(query_text)['Id'].index[0]

            #print(index_train,index_test)

            test_data.at[index_test, 'ConfirmedPred'] = train_data.loc[index_train,'ConfirmedCases']

        except IndexError:

            pass

        this_date = this_date + np.timedelta64(1,'D')

    last_confirmed = data_this_state.tail(1)['ConfirmedCases'].to_numpy()[0]

    this_date = last_date + np.timedelta64(1,'D')

    while this_date <= last_day:

        query_text = "Country == '{0}' & State == '{1}' & Date == '{2}'".format(country, state, this_date)

        index = test_data.query(query_text)['ForecastId'].index[0]

        this_confirmed = last_confirmed * avg_growth_rate

        this_fatality = this_confirmed * avg_fatality_rate

        #print(this_date,this_confirmed,this_fatality)

        test_data.at[index, 'ConfirmedPred'] = this_confirmed

        test_data.at[index, 'FatalitiesPred'] = this_fatality

        this_date = this_date + np.timedelta64(1, 'D')

        last_confirmed = this_confirmed



large_countries = train_data['Country'].unique()



country_confirmed_parameters = {}



for i_country in large_countries:

    print(i_country)

    data_this_country = train_data.query("Country == '{0}'".format(i_country))

    states = data_this_country.State.unique()

    for i_state in states:

        data_this_state = data_this_country.query("State == '{0}'".format(i_state))

        #print(data_this_state)

        data_this_state_100 = data_this_state.query("ConfirmedCases > 100 & Fatalities > 0")

        if data_this_state_100.shape[0] < 2:

            predictions_100minus(i_country, i_state, data_this_state)

        else:

            confirmed_parameters_this_state = do_fitting(data_this_state_100)

            first_day = data_this_state['Date'].iloc[0].to_numpy().astype('datetime64[D]')

            first_day_100 = data_this_state_100['Date'].iloc[0].to_numpy().astype('datetime64[D]')

            confirmed_parameters_this_state = confirmed_parameters_this_state + (first_day, first_day_100)

            state_key = '{0}; {1}'.format(i_country, i_state)

            country_confirmed_parameters[state_key] = confirmed_parameters_this_state

            predictions_100plus(state_key,country_confirmed_parameters,plot_it=False)





from ipywidgets import widgets, interactive



i_country = widgets.Dropdown(options=train_data['Country'].unique().tolist(),description='Country:',value='Iran')

i_state = widgets.Dropdown(description='State')

#i_state = widgets.Dropdown(options=train_data['State'].unique().tolist(),description='State:')



def update(*args):

    i_state.options = train_data.query("Country == '{0}'".format(i_country.value)).State.unique()

    

i_country.observe(update)



def plotit(i_country,i_state):

    try:

        state_key = '{0}; {1}'.format(i_country, i_state)

        if state_key == 'Iran; None':

            predictions_100plus('Iran; All',country_confirmed_parameters)

        else:

            predictions_100plus(state_key,country_confirmed_parameters)

    

    except KeyError:

        print('This Country/State combination has less than 100 cases')

    



interactive(plotit,i_country=i_country,i_state=i_state)

list_confirmed_parameters = list(country_confirmed_parameters.values())



slopes = []

offsets = []



for i_element in list_confirmed_parameters:

    slopes.append(i_element[4])

    offsets.append(i_element[6])



average_slope = np.array(slopes).mean()

average_offset = np.array(offsets).mean()

prediction = test_data[['ForecastId','ConfirmedPred','FatalitiesPred']]



prediction.to_csv('submission.csv',

                  header=['ForecastId','ConfirmedCases','Fatalities'],index=False)
