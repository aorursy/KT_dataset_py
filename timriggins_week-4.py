import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from tqdm.notebook import tqdm

from scipy.optimize import curve_fit
!git clone https://github.com/CSSEGISandData/COVID-19 -b web-data COVID-19-new
train = pd.read_csv('../input/covid19-global-forecasting-week-4/train.csv')

test = pd.read_csv('../input/covid19-global-forecasting-week-4/test.csv')



# today_countries_state = pd.read_csv("COVID-19-new/data/cases_state.csv")

# today_countries = pd.read_csv("COVID-19-new/data/cases_country.csv")



train['State_Country'] = [s + '_' + c if s == s else c for s,c in train[['Province_State', 'Country_Region']].values ]

# today_countries_state['State_Country'] = [s + '_' + c if s == s else c for s,c in today_countries_state[['Province_State', 'Country_Region']].values ]

test['State_Country'] = [s + '_' + c if s == s else c for s,c in test[['Province_State', 'Country_Region']].values ]



train.loc[(train['Date']=='2020-03-24')&(train['State_Country']=='France'),'ConfirmedCases'] = 22654

train.loc[(train['Date']=='2020-03-24')&(train['State_Country']=='France'),'Fatalities'] = 1000



# non_state_country = train[train.Province_State.isnull()].Country_Region.unique()

# today_countries = today_countries[today_countries.Country_Region.isin(non_state_country)][['Country_Region', 'Confirmed', 'Deaths']]

# today_countries.columns = ['Country_Region', 'ConfirmedCases', 'Fatalities']

# today_countries['Date'] = '2020-04-15'

# train = pd.concat([train, today_countries], sort = False)



# today_countries_state = today_countries_state[today_countries_state.State_Country.isin(train.State_Country.unique())][['Province_State',

#                                                                                                 'Country_Region', 'Confirmed', 'Deaths']]

# today_countries_state.columns = ['Province_State', 'Country_Region', 'ConfirmedCases', 'Fatalities']

# today_countries_state['Date'] = '2020-04-15'

# train = pd.concat([train, today_countries_state], sort = False)



train['State_Country'] = [s + '_' + c if s == s else c for s,c in train[['Province_State', 'Country_Region']].values ]

train = train.sort_values(['State_Country', 'Date']).reset_index(drop = True)



train['Id'] = train.index
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
today_countries_state
def predict(data, country, len_predict, metrics, len_intersection, bound_0, bound_1):

    country_data = data[data['State_Country']==country]

    

    if metrics != 'Fatalities':

        if country_data[metrics].values.max() > 5000:

            start_people = 3

        elif country_data[metrics].values.max() > 500:

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

#             y_max = max(country_data['Population'].values[0] * 0.001, y_data[-1] * 5)

            y_max = y_data[-1] * 10

        else:

            y_max = y_data[-1] * 7

        y_min = y_data[-1]

        if metrics != 'Fatalities':

            diff_k = max(1,  y_data[-1] - y_data[-2])

        else:

            diff_k = 1



        best_score = 100000000000



        if len(x_data) < 10:

            range_num = 1

        else:

            range_num = len(x_data) - 9

            

        for i in range(range_num):

            popt, pcov = curve_fit(log_curve, x_data[i:], y_data[i:],  bounds=([bound_0 - 0.05, bound_1 - 7.5 , y_min, 0 ],

                                                                  [bound_0 + 0.1, bound_1 + 7.5, y_max, diff_k]),

                                p0=[bound_0, bound_1 ,(y_min + y_max) / 2, 0], maxfev=100000)

            

            tmp_pred = log_curve(x_data[-5:], popt[0], popt[1], popt[2], popt[3])

            score = rmsle(y_data[-5:], tmp_pred)

            if score < best_score:

                best_score = score

                best_i = i

                best_popt = popt

        

        

#         if y_min < 10000:

#             popt, pcov = curve_fit(log_curve, x_data, y_data,bounds=([bound_0 - 0.05, bound_1 - 7.5 , y_min, 0 ],

#                                                                   [bound_0 + 0.1, bound_1 + 7.5, y_max, diff_k]), 

#                             p0=[bound_0, bound_1 ,(y_min + y_max) / 2, 0], maxfev=100000)

#         else:

#             popt, pcov = curve_fit(log_curve, x_data, y_data,bounds=([0, 0 , y_min, 0 ],

#                                                                   [np.inf,np.inf, y_max, diff_k]), 

#                             p0=[bound_0, bound_1 ,(y_min + y_max) / 2, 0], maxfev=100000)

            

        y_val = np.arange(len(x_data) - len_intersection, len(x_data) + len_predict - len_intersection)

#         print(x_data)

#         print(y_data)

#         print([bound_0, bound_1 ,(y_min + y_max) / 2, 0])

#         print([0, 0, y_min, 0 ],

#                                                                   [np.inf, np.inf, y_max, diff_k])        

#         print(y_val)

        return  popt, log_curve(y_val, best_popt[0], best_popt[1], best_popt[2], best_popt[3])

    





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

    for case in [1, 30, 100, 500]:

        try:

            dict_case_date[country] += [np.where(dict_values[country] >= case)[0][0]]

        except:

            dict_case_date[country] += [-1]

    dict_case_date[country] = np.array(dict_case_date[country])

    

dict_predict = {}



for country in tqdm(train.State_Country.unique()):



    popt, pred_values = predict(data_train, country, len_predict, metrics, len_intersection, 0.15, 30)

    

    if pred_values[len_intersection] < train[train.State_Country == country][metrics].values[-1]:

        pred_values += ( train[train.State_Country == country][metrics].values[-1] - pred_values[len_intersection] )

    

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



for country in tqdm(train.State_Country.unique()):



    popt, pred_values = predict(data_train, country, len_predict, metrics, len_intersection, 0.15, 30)

    

    if pred_values[len_intersection] < train[train.State_Country == country][metrics].values[-1]:

        pred_values += ( train[train.State_Country == country][metrics].values[-1] - pred_values[len_intersection] )

        

    dict_predict[country] = pred_values



test[metrics] = 0

for country in test['State_Country'].unique():

    test.loc[test.State_Country == country, metrics] = dict_predict[country]
test1 = pd.merge(test1, tmin[['State_Country','diff_1_cc','diff_1_ft']], on='State_Country')

test2 = pd.merge(test2, tmax[['State_Country','diff_1_cc','diff_1_ft']], on='State_Country')



test1['ConfirmedCases'] = test1['diff_1_cc']*1.2

test1['ConfirmedCases'] = test1.groupby(['State_Country'])['ConfirmedCases'].cumsum()

test1['Fatalities'] = test1['diff_1_ft']*1.2

test1['Fatalities'] = test1.groupby(['State_Country'])['Fatalities'].cumsum()



test2['ConfirmedCases'] = test2['diff_1_cc']

test2['Fatalities'] = test2['diff_1_ft']
def make_simple(train, test):

    

    train = pd.read_csv('../input/covid19-global-forecasting-week-4/train.csv')

    test = pd.read_csv('../input/covid19-global-forecasting-week-4/test.csv')



#     today_countries_state = pd.read_csv("COVID-19-new/data/cases_state.csv")

#     today_countries = pd.read_csv("COVID-19-new/data/cases_country.csv")



    train['State_Country'] = [s + '_' + c if s == s else c for s,c in train[['Province_State', 'Country_Region']].values ]

#     today_countries_state['State_Country'] = [s + '_' + c if s == s else c for s,c in today_countries_state[['Province_State', 'Country_Region']].values ]

    test['State_Country'] = [s + '_' + c if s == s else c for s,c in test[['Province_State', 'Country_Region']].values ]



    train.loc[(train['Date']=='2020-03-24')&(train['State_Country']=='France'),'ConfirmedCases'] = 22654

    train.loc[(train['Date']=='2020-03-24')&(train['State_Country']=='France'),'Fatalities'] = 1000



#     non_state_country = train[train.Province_State.isnull()].Country_Region.unique()

#     today_countries = today_countries[today_countries.Country_Region.isin(non_state_country)][['Country_Region', 'Confirmed', 'Deaths']]

#     today_countries.columns = ['Country_Region', 'ConfirmedCases', 'Fatalities']

#     today_countries['Date'] = '2020-04-15'

#     train = pd.concat([train, today_countries], sort = False)



#     today_countries_state = today_countries_state[today_countries_state.State_Country.isin(train.State_Country.unique())][['Province_State',

#                                                                                                     'Country_Region', 'Confirmed', 'Deaths']]

#     today_countries_state.columns = ['Province_State', 'Country_Region', 'ConfirmedCases', 'Fatalities']

#     today_countries_state['Date'] = '2020-04-15'

#     train = pd.concat([train, today_countries_state], sort = False)



    train['State_Country'] = [s + '_' + c if s == s else c for s,c in train[['Province_State', 'Country_Region']].values ]

    train = train.sort_values(['State_Country', 'Date']).reset_index(drop = True)



    train['Id'] = train.index



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

            

    train['shift_1_cc'] = train.groupby(['State_Country'])['ConfirmedCases'].shift(3)

    train['shift_1_ft'] = train.groupby(['State_Country'])['Fatalities'].shift(3)



    train['diff_1_cc'] = (train['ConfirmedCases']-train['shift_1_cc'] )/3 * 0.95

    train['diff_1_ft'] = (train['Fatalities']-train['shift_1_ft'] )/3 * 0.95



    tmin = train[train['Date']==test['Date'].min()]

    tmax = train[train['Date']==train['Date'].max()]



    test1 = test[test['Date']<=train['Date'].max()]

    test2 = test[test['Date']>train['Date'].max()]



    test1 = pd.merge(test1, tmin[['State_Country','diff_1_cc','diff_1_ft']], on='State_Country')

    test2 = pd.merge(test2, tmax[['State_Country','diff_1_cc','diff_1_ft']], on='State_Country')



    test1['ConfirmedCases'] = test1['diff_1_cc']*1.2

    test1['ConfirmedCases'] = test1.groupby(['State_Country'])['ConfirmedCases'].cumsum()

    test1['Fatalities'] = test1['diff_1_ft']*1.2

    test1['Fatalities'] = test1.groupby(['State_Country'])['Fatalities'].cumsum()



    test2['ConfirmedCases'] = test2['diff_1_cc']

    test2['Fatalities'] = test2['diff_1_ft']



    test2.loc[test2['Date']==test2['Date'].min(),'ConfirmedCases'] = list(train.loc[train['Date']==test1['Date'].max(),'ConfirmedCases'])

    test2.loc[test2['Date']==test2['Date'].min(),'Fatalities'] = list(train.loc[train['Date']==test1['Date'].max(),'Fatalities'])



    test2['ConfirmedCases'] = test2.groupby(['State_Country'])['ConfirmedCases'].cumsum()

    test2['Fatalities'] = test2.groupby(['State_Country'])['Fatalities'].cumsum()



    test_final = test1.append(test2).reset_index(drop = True)

#     test_final = test_final[test_final.Date >= '2020-04-09']

    return test_final



new_test_s = make_simple(train, test)
test['New_Fatalities'] = test['ForecastId'].map(dict(zip(new_test_s['ForecastId'], new_test_s['Fatalities'])))

test['New_Fatalities'] = (test['New_Fatalities'] * 0.5  + test['Fatalities'] * 0.5)



test['New_ConfirmedCases'] = test['ForecastId'].map(dict(zip(new_test_s['ForecastId'], new_test_s['ConfirmedCases'])))

test['New_ConfirmedCases'] = (test['New_ConfirmedCases'] * 0.5 + test['ConfirmedCases'] * 0.5 )
test['New_Fatalities'] = test['New_Fatalities'].clip(0, test['New_ConfirmedCases'] * 0.2)
submit = pd.read_csv('../input/covid19-global-forecasting-week-4/submission.csv')

submit['Fatalities'] = test['New_Fatalities'].astype('float')

submit['ConfirmedCases'] = test['New_ConfirmedCases'].astype('float')

submit.to_csv('submission.csv',index=False)