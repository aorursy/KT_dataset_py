import numpy as np

import pandas as pd

import lightgbm as lgb

import matplotlib.pyplot as plt

from scipy.optimize import curve_fit
train = pd.read_csv('../input/covid19-global-forecasting-week-2/train.csv')

test = pd.read_csv('../input/covid19-global-forecasting-week-2/test.csv')

world_population = pd.read_csv('../input/population-by-country-2020/population_by_country_2020.csv')
world_population = world_population[['Country (or dependency)', 'Population (2020)']]

world_population.columns = ['Country (or dependency)', 'Population']

world_population.loc[world_population['Country (or dependency)']=='United States', 'Country (or dependency)'] = 'US'



train = train.merge(world_population, left_on='Country_Region', right_on='Country (or dependency)', how='left')

test = test.merge(world_population, left_on='Country_Region', right_on='Country (or dependency)', how='left')
train['State_Country'] = [s + '_' + c if s == s else c for s,c in train[['Province_State', 'Country_Region']].values ]

test['State_Country'] = [s + '_' + c if s == s else c for s,c in test[['Province_State', 'Country_Region']].values ]
train.loc[(train['Date']=='2020-03-24')&(train['State_Country']=='France'),'ConfirmedCases'] = 22654

train.loc[(train['Date']=='2020-03-24')&(train['State_Country']=='France'),'Fatalities'] = 1000
for metric in ['ConfirmedCases', 'Fatalities']:

    dict_values = train.groupby('State_Country')[metric].apply(np.array).to_dict()



    for country in dict_values:

        if sum(np.diff(dict_values[country]) < 0):

            print(country, metric)

            new_val = [dict_values[country][-1]]

            for val_1, val_2 in zip(dict_values[country][1:][::-1], dict_values[country][:-1][::-1]):

                if val_2 <= new_val[-1]:

                    new_val += [val_2]

                else:

                    new_val += [new_val[-1]]

            new_val = np.array(new_val[::-1])

            train.loc[train.State_Country == country, metric] = new_val
train['max_case'] = train['State_Country'].map( train.groupby('State_Country').ConfirmedCases.max() )

train['pct_c'] = train.ConfirmedCases.pct_change()
def predict(data, country, len_predict, metrics, len_intersection):

    country_data = data[data['State_Country']==country]

    

    if country_data[metrics].max() <= 50:

        start = 0

    else:

        start = 1

    country_data = country_data.iloc[dict_case_date[country][start]:, :]



    x_data = range(len(country_data.index))

    y_data = country_data[metrics].values



    if len(x_data) <= 1:

        y_min = 0

        y_max = 100

        

        y_val = np.arange(len(x_data), len(x_data) + len_predict)

        

        k = 100 if metrics == 'ConfirmedCases' else 2

        return [-1, -1, -1], log_curve(y_val, 0.3, 30, 2)

#         return [-1, -1, -1], np.ones(len_predict)



    else:

        add_min = (1 + country_data.pct_c.values[-3:].mean()) ** (1 / 2)

        add_max = (1 + country_data.pct_c.values[-3:].max()) ** (1 / 2)



        day_left = max(10, 70 - len(x_data))



        y_min = y_data[-1] * ( add_min ** day_left )

        k = 0.05 if metrics == 'ConfirmedCases' else 0.05 * 0.01

        if y_min > country_data['Population'].values[0] * k:

            y_min = country_data['Population'].values[0] * k

        y_max = y_data[-1] * ( add_max ** day_left + 30 )

        k = 0.15 if metrics == 'ConfirmedCases' else 0.15 * 0.01

        if y_max > country_data['Population'].values[0] * k:

            y_max = country_data['Population'].values[0] * k

        if add_min == add_max:

            y_min = y_data[-1]

            y_max = y_data[-1] * ( 1.3 ** 30 )







        if y_data.max() < 100 and metrics == 'ConfirmedCases':

            y_min = 0

            y_max = 50000

        if y_data.max() < 5 and metrics == 'Fatalities':

            y_min = 0

            y_max = 100

            

        if y_max > 250000 and metrics == 'ConfirmedCases':

            y_min = 1000

            y_max = 250000



        if y_max > 20000 and metrics == 'Fatalities':

            y_min = 100

            y_max = 20000

#     print(y_min, y_max)

        popt, pcov = curve_fit(log_curve, x_data, y_data, bounds=([0.1, 10, y_min ],[0.35, 50, y_max]), 

                            p0=[0.2,30,(y_min + y_max) / 2], maxfev=10000)



        y_val = np.arange(len(x_data) - len_intersection, len(x_data) + len_predict - len_intersection)



        return  popt, log_curve(y_val, popt[0], popt[1], popt[2])





def log_curve(x, k, x_0, ymax):

    return ymax / (1 + np.exp(-k*(x-x_0)))
metrics = 'ConfirmedCases'

dict_values = train.groupby('State_Country')[metrics].apply(np.array).to_dict()

dict_case_date = {}

for country in dict_values:

    dict_case_date[country] = []

    for case in [1, 10, 100, 250, 500, 1000, 2500, 5000]:

        try:

            dict_case_date[country] += [np.where(dict_values[country] >= case)[0][0]]

        except:

            dict_case_date[country] += [-1]

    dict_case_date[country] = np.array(dict_case_date[country])

    

    

dict_predict = {}



data_train = train.copy()

data_val = test.copy()



len_predict = data_val[data_val.State_Country == country].shape[0]



len_intersection = len(set(data_train.Date.unique()) & set(data_val.Date.unique()))



for country in train.State_Country.unique():

    

    if data_train[data_train.State_Country == country][metrics].max() <= 10:

        popt, pred_values = predict(data_train, country, len_predict, metrics, len_intersection)

    else:

        popt, pred_values = predict(data_train, country, len_predict, metrics, len_intersection)

    dict_predict[country] = pred_values
test['ConfirmedCases'] = 0

for country in test['State_Country'].unique():

    test.loc[test.State_Country == country, 'ConfirmedCases'] = dict_predict[country]
metrics = 'Fatalities'

dict_values = train.groupby('State_Country')[metrics].apply(np.array).to_dict()

dict_case_date = {}

for country in dict_values:

    dict_case_date[country] = []

    for case in [1, 10, 100, 250, 500, 1000, 2500, 5000]:

        try:

            dict_case_date[country] += [np.where(dict_values[country] >= case)[0][0]]

        except:

            dict_case_date[country] += [-1]

    dict_case_date[country] = np.array(dict_case_date[country])

    

    

dict_predict = {}



data_train = train.copy()

data_val = test.copy()



len_predict = data_val[data_val.State_Country == country].shape[0]



len_intersection = len(set(data_train.Date.unique()) & set(data_val.Date.unique()))



for country in train.State_Country.unique():

    

    if data_train[data_train.State_Country == country][metrics].max() <= 3:

        popt, pred_values = predict(data_train, country, len_predict, metrics, len_intersection)

    else:

        popt, pred_values = predict(data_train, country, len_predict, metrics, len_intersection)

    dict_predict[country] = pred_values
test['Fatalities'] = 0

for country in test['State_Country'].unique():

    test.loc[test.State_Country == country, 'Fatalities'] = dict_predict[country]
submit = pd.read_csv('../input/covid19-global-forecasting-week-2/submission.csv')

submit['Fatalities'] = test['Fatalities'].astype('float')

submit['ConfirmedCases'] = test['ConfirmedCases'].astype('float')

submit.to_csv('submission.csv',index=False)
def rmsle(true, pred):

    true = np.array(true)

    pred = np.array(pred)

    return np.mean((np.log1p(true) - np.log1p(pred)) ** 2) ** (1/2)
new_test = test.merge(train[['State_Country', 'Date', 'Fatalities']], on = ['State_Country', 'Date'])

rmsle(new_test['Fatalities_y'], new_test['Fatalities_x'])
new_test = test.merge(train[['State_Country', 'Date', 'ConfirmedCases']], on = ['State_Country', 'Date'])

rmsle(new_test['ConfirmedCases_y'], new_test['ConfirmedCases_x'])