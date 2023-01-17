import scipy

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from scipy.optimize import curve_fit

train = pd.read_csv('../input/covid19-global-forecasting-week-4/train.csv')

test = pd.read_csv('../input/covid19-global-forecasting-week-4/test.csv')
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
def predict(data, country, len_predict, metrics, len_intersection, bound_0, bound_1):

    country_data = data[data['State_Country']==country]

    

    if metrics != 'Fatalities':

        if country_data[metrics].values.max() > 500:

            start_people = 2

        else:

            start_people = 0

    else:

        if country_data[metrics].values.max() > 50:

            start_people = 1

        else:

            start_people = 0        



    country_data = country_data.iloc[dict_case_date[country][start_people]:, :]



    x_data = range(len(country_data.index))

    y_data = country_data[metrics].values



    if len(x_data) <= 1:

        y_val = np.arange(len(x_data), len(x_data) + len_predict)

        if metrics != 'Fatalities':

            return [-1, -1, -1], log_curve(y_val, bound_0, bound_1, 100, 1)

        else:

            return [-1, -1, -1], log_curve(y_val, bound_0, bound_1, 3, 0)            

    else:

        if metrics != 'Fatalities':

            y_max = y_data[-1] * 15

        else:

            y_max = y_data[-1] * 10

        y_min = y_data[-1]

        

        if metrics != 'Fatalities':

            diff_k = max(1,  y_data[-1] - y_data[-2])

        else:

            diff_k = 1

        

        popt, pcov = curve_fit(log_curve, x_data, y_data,bounds=([bound_0 - 0.05, bound_1 - 7.5 , y_min, 0 ],

                                                                  [bound_0 + 0.1, bound_1 + 7.5, y_max, diff_k]), 

                            p0=[bound_0, bound_1 ,(y_min + y_max) / 2, 0], maxfev=100000)



        y_val = np.arange(len(x_data) - len_intersection, len(x_data) + len_predict - len_intersection)

#         print(x_data)

#         print(y_data)

#         print([bound_0, bound_1 ,(y_min + y_max) / 2, 0])

#         print([0, 0, y_min, 0 ],

#                                                                   [np.inf, np.inf, y_max, diff_k])        

#         print(y_val)

        return  popt, log_curve(y_val, popt[0], popt[1], popt[2], popt[3])

    





def log_curve(x, k, x_0, ymax, x_1):

    return ymax / (1 + np.exp(-k*(x-x_0))) + x_1 * x



def rmsle(true, pred):

    true = np.array(true)

    pred = np.array(pred)

    return np.mean((np.log1p(true) - np.log1p(pred)) ** 2) ** (1/2)
metrics = 'ConfirmedCases'



data_train = train.copy()

data_val = test.copy()

len_predict = data_val[data_val.State_Country == country].shape[0]

len_intersection = len(set(data_train.Date.unique()) & set(data_val.Date.unique()))



dict_values = data_train.groupby('State_Country')[metrics].apply(np.array).to_dict()

dict_case_date = {}

for country in dict_values:

    dict_case_date[country] = []

    for case in [1, 10, 100]:

        try:

            dict_case_date[country] += [np.where(dict_values[country] >= case)[0][0]]

        except:

            dict_case_date[country] += [-1]

    dict_case_date[country] = np.array(dict_case_date[country])

    

dict_predict = {}



for country in train.State_Country.unique():



    popt, pred_values = predict(data_train, country, len_predict, metrics, len_intersection, 0.15, 30)

    dict_predict[country] = pred_values



test[metrics] = 0

for country in test['State_Country'].unique():

    test.loc[test.State_Country == country, metrics] = dict_predict[country]
metrics = 'Fatalities'



dict_values = data_train.groupby('State_Country')[metrics].apply(np.array).to_dict()

dict_case_date = {}

for country in dict_values:

    dict_case_date[country] = []

    for case in [1, 5]:

        try:

            dict_case_date[country] += [np.where(dict_values[country] >= case)[0][0]]

        except:

            dict_case_date[country] += [-1]

    dict_case_date[country] = np.array(dict_case_date[country])

    

dict_predict = {}



for country in train.State_Country.unique():



    popt, pred_values = predict(data_train, country, len_predict, metrics, len_intersection, 0.15, 30)

    dict_predict[country] = pred_values



test[metrics] = 0

for country in test['State_Country'].unique():

    test.loc[test.State_Country == country, metrics] = dict_predict[country]
submit = pd.read_csv('../input/covid19-global-forecasting-week-4/submission.csv')

submit['Fatalities'] = test['Fatalities'].astype('float')

submit['ConfirmedCases'] = test['ConfirmedCases'].astype('float')

submit.to_csv('submission.csv',index=False)
test[test.State_Country == 'Alaska_US']
len_intersection