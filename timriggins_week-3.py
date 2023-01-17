import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from tqdm.notebook import tqdm

from scipy.optimize import curve_fit
train = pd.read_csv('../input/covid19-global-forecasting-week-3/train.csv')

test = pd.read_csv('../input/covid19-global-forecasting-week-3/test.csv')

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
def clear_population(train):

    train.loc[train.State_Country == 'Czechia', 'Population'] = 10704146

    train.loc[train.State_Country == 'Korea, South', 'Population'] =  51259914

    train.loc[train.State_Country == 'Saint Vincent and the Grenadines', 'Population'] =  110210 

    train.loc[train.State_Country == 'Sao Tome and Principe', 'Population'] =  211028  

    train.loc[train.State_Country == 'West Bank and Gaza', 'Population'] =  4569000  

    train.loc[train.State_Country == 'Diamond Princess', 'Population'] =  2670  



    train.loc[train.State_Country == 'Kosovo', 'Population'] = 1810366

    train.loc[train.State_Country == 'MS Zaandam', 'Population'] = 1432

    train.loc[train.State_Country == 'Taiwan*', 'Population'] = 23816775

    train.loc[train.State_Country == 'Saint Kitts and Nevis', 'Population'] = 53106

    train.loc[train.State_Country == 'Burma', 'Population'] = 54409800

    train.loc[train.State_Country == 'Kosovo', 'Population'] = 1810366

    train.loc[train.State_Country == 'Congo (Brazzaville)', 'Population'] = 5175324

    train.loc[train.State_Country == 'Congo (Kinshasa)', 'Population'] = 14342000

    train.loc[train.State_Country == "Cote d'Ivoire", 'Population'] = 26207028



    train.loc[train.State_Country == 'Australian Capital Territory_Australia', 'Population'] = 418440

    train.loc[train.State_Country == 'New South Wales_Australia', 'Population'] = 8023700

    train.loc[train.State_Country == 'Northern Territory_Australia', 'Population'] = 249220

    train.loc[train.State_Country == 'Queensland_Australia', 'Population'] = 5110000

    train.loc[train.State_Country == 'South Australia_Australia', 'Population'] = 1700000

    train.loc[train.State_Country == 'Tasmania_Australia', 'Population'] = 524170 

    train.loc[train.State_Country == 'Victoria_Australia', 'Population'] = 6490000 

    train.loc[train.State_Country == 'Western Australia_Australia', 'Population'] = 2760000 



    train.loc[train.State_Country == 'Alberta_Canada', 'Population'] = 4413146

    train.loc[train.State_Country == 'British Columbia_Canada', 'Population'] = 5110917

    train.loc[train.State_Country == 'Manitoba_Canada', 'Population'] = 1377517

    train.loc[train.State_Country == 'New Brunswick_Canada', 'Population'] = 761214

    train.loc[train.State_Country == 'Newfoundland and Labrador_Canada', 'Population'] = 536000

    train.loc[train.State_Country == 'Northwest Territories_Canada', 'Population'] = 44904 

    train.loc[train.State_Country == 'Nova Scotia_Canada', 'Population'] = 977457 

    train.loc[train.State_Country == 'Ontario_Canada', 'Population'] = 14711827 

    train.loc[train.State_Country == 'Prince Edward Island_Canada', 'Population'] = 158158 

    train.loc[train.State_Country == 'Quebec_Canada', 'Population'] = 8537674 

    train.loc[train.State_Country == 'Saskatchewan_Canada', 'Population'] = 1181666 

    train.loc[train.State_Country == 'Yukon_Canada', 'Population'] = 41078 



    train.loc[train.State_Country == 'Faroe Islands_Denmark', 'Population'] = 48863

    train.loc[train.State_Country == 'Greenland_Denmark', 'Population'] = 56746

    train.loc[train.State_Country == 'Denmark', 'Population'] = 5792202



    train.loc[train.State_Country == 'French Guiana_France', 'Population'] = 290691

    train.loc[train.State_Country == 'French Polynesia_France', 'Population'] = 280908

    train.loc[train.State_Country == 'Guadeloupe_France', 'Population'] = 400124

    train.loc[train.State_Country == 'Martinique_France', 'Population'] = 375335

    train.loc[train.State_Country == 'Mayotte_France', 'Population'] = 272815

    train.loc[train.State_Country == 'New Caledonia_France', 'Population'] = 285498

    train.loc[train.State_Country == 'Reunion_France', 'Population'] = 859959

    train.loc[train.State_Country == 'Saint Barthelemy_France', 'Population'] = 9870

    train.loc[train.State_Country == 'St Martin_France', 'Population'] = 38500

    train.loc[train.State_Country == 'France', 'Population'] = 65273511



    train.loc[train.State_Country == 'Aruba_Netherlands', 'Population'] = 106766

    train.loc[train.State_Country == 'Curacao_Netherlands', 'Population'] = 164093

    train.loc[train.State_Country == 'Sint Maarten_Netherlands', 'Population'] = 42754

    train.loc[train.State_Country == 'Netherlands', 'Population'] = 17134872 



    train.loc[train.State_Country == 'Anguilla_United Kingdom', 'Population'] = 14969

    train.loc[train.State_Country == 'Bermuda_United Kingdom', 'Population'] = 62333

    train.loc[train.State_Country == 'British Virgin Islands_United Kingdom', 'Population'] = 30231

    train.loc[train.State_Country == 'Cayman Islands_United Kingdom', 'Population'] = 65528 

    train.loc[train.State_Country == 'Channel Islands_United Kingdom', 'Population'] = 173863 

    train.loc[train.State_Country == 'Gibraltar_United Kingdom', 'Population'] = 33691 

    train.loc[train.State_Country == 'Isle of Man_United Kingdom', 'Population'] = 91556 

    train.loc[train.State_Country == 'Montserrat_United Kingdom', 'Population'] = 4991 

    train.loc[train.State_Country == 'Turks and Caicos Islands_United Kingdom', 'Population'] = 38717 

    train.loc[train.State_Country == 'United Kingdom', 'Population'] = 67886011 



    train.loc[train.State_Country == 'Anhui_China', 'Population'] = 62548000

    train.loc[train.State_Country == 'Beijing_China', 'Population'] = 21707000

    train.loc[train.State_Country == 'Chongqing_China', 'Population'] = 30752000

    train.loc[train.State_Country == 'Fujian_China', 'Population'] = 39110000 

    train.loc[train.State_Country == 'Gansu_China', 'Population'] = 26257000

    train.loc[train.State_Country == 'Guangdong_China', 'Population'] = 111690000

    train.loc[train.State_Country == 'Guangxi_China', 'Population'] = 48850000

    train.loc[train.State_Country == 'Guizhou_China', 'Population'] = 35800000 

    train.loc[train.State_Country == 'Hainan_China', 'Population'] = 9258000

    train.loc[train.State_Country == 'Hebei_China', 'Population'] = 75195000

    train.loc[train.State_Country == 'Heilongjiang_China', 'Population'] = 37887000

    train.loc[train.State_Country == 'Henan_China', 'Population'] = 95590000 

    train.loc[train.State_Country == 'Hong Kong_China', 'Population'] = 7496981

    train.loc[train.State_Country == 'Hubei_China', 'Population'] = 59020000

    train.loc[train.State_Country == 'Hunan_China', 'Population'] = 68602000

    train.loc[train.State_Country == 'Inner Mongolia_China', 'Population'] = 25286000

    train.loc[train.State_Country == 'Jiangxi_China', 'Population'] = 80293000

    train.loc[train.State_Country == 'Jiangsu_China', 'Population'] = 46221000

    train.loc[train.State_Country == 'Jilin_China', 'Population'] = 27174000

    train.loc[train.State_Country == 'Liaoning_China', 'Population'] = 43689000

    train.loc[train.State_Country == 'Macau_China', 'Population'] = 647097

    train.loc[train.State_Country == 'Ningxia_China', 'Population'] = 6818000

    train.loc[train.State_Country == 'Qinghai_China', 'Population'] = 5984000

    train.loc[train.State_Country == 'Shaanxi_China', 'Population'] = 38354000

    train.loc[train.State_Country == 'Shandong_China', 'Population'] = 100058000

    train.loc[train.State_Country == 'Shanghai_China', 'Population'] = 24183000

    train.loc[train.State_Country == 'Shanxi_China', 'Population'] = 37024000

    train.loc[train.State_Country == 'Sichuan_China', 'Population'] = 83020000

    train.loc[train.State_Country == 'Tianjin_China', 'Population'] = 15569000

    train.loc[train.State_Country == 'Tibet_China', 'Population'] = 3371000

    train.loc[train.State_Country == 'Xinjiang_China', 'Population'] = 24467000

    train.loc[train.State_Country == 'Yunnan_China', 'Population'] = 48005000

    train.loc[train.State_Country == 'Zhejiang_China', 'Population'] = 56570000



    train.loc[train.State_Country == 'Alabama_US', 'Population'] = 4908621

    train.loc[train.State_Country == 'Alaska_US', 'Population'] = 734002

    train.loc[train.State_Country == 'Arizona_US', 'Population'] = 7378494

    train.loc[train.State_Country == 'Arkansas_US', 'Population'] = 3038999

    train.loc[train.State_Country == 'California_US', 'Population'] = 39937489

    train.loc[train.State_Country == 'Colorado_US', 'Population'] = 5845526

    train.loc[train.State_Country == 'Connecticut_US', 'Population'] = 3563077

    train.loc[train.State_Country == 'Delaware_US', 'Population'] = 982895

    train.loc[train.State_Country == 'District of Columbia_US', 'Population'] = 720687

    train.loc[train.State_Country == 'Florida_US', 'Population'] = 21992985

    train.loc[train.State_Country == 'Georgia_US', 'Population'] = 10736059

    train.loc[train.State_Country == 'Guam_US', 'Population'] = 168775

    train.loc[train.State_Country == 'Hawaii_US', 'Population'] = 1412687

    train.loc[train.State_Country == 'Idaho_US', 'Population'] = 1826156

    train.loc[train.State_Country == 'Illinois_US', 'Population'] = 12659682

    train.loc[train.State_Country == 'Indiana_US', 'Population'] = 6745354

    train.loc[train.State_Country == 'Iowa_US', 'Population'] = 3179849

    train.loc[train.State_Country == 'Kansas_US', 'Population'] = 2910357

    train.loc[train.State_Country == 'Kentucky_US', 'Population'] = 4499692

    train.loc[train.State_Country == 'Louisiana_US', 'Population'] = 4645184

    train.loc[train.State_Country == 'Maine_US', 'Population'] = 1345790

    train.loc[train.State_Country == 'Maryland_US', 'Population'] = 6083116

    train.loc[train.State_Country == 'Massachusetts_US', 'Population'] = 6976597

    train.loc[train.State_Country == 'Michigan_US', 'Population'] = 10045029

    train.loc[train.State_Country == 'Minnesota_US', 'Population'] = 5700671

    train.loc[train.State_Country == 'Mississippi_US', 'Population'] = 2989260

    train.loc[train.State_Country == 'Missouri_US', 'Population'] = 6169270

    train.loc[train.State_Country == 'Montana_US', 'Population'] = 1086759

    train.loc[train.State_Country == 'Nebraska_US', 'Population'] = 1952570

    train.loc[train.State_Country == 'Nevada_US', 'Population'] = 3139658

    train.loc[train.State_Country == 'New Hampshire_US', 'Population'] = 1371246

    train.loc[train.State_Country == 'New Jersey_US', 'Population'] = 8936574

    train.loc[train.State_Country == 'New Mexico_US', 'Population'] = 2096640

    train.loc[train.State_Country == 'New York_US', 'Population'] = 19440469

    train.loc[train.State_Country == 'North Carolina_US', 'Population'] = 10611862

    train.loc[train.State_Country == 'North Dakota_US', 'Population'] = 761723

    train.loc[train.State_Country == 'Ohio_US', 'Population'] = 11747694

    train.loc[train.State_Country == 'Oklahoma_US', 'Population'] = 3954821

    train.loc[train.State_Country == 'Oregon_US', 'Population'] = 4301089

    train.loc[train.State_Country == 'Pennsylvania_US', 'Population'] = 12820878

    train.loc[train.State_Country == 'Puerto Rico_US', 'Population'] = 3032165

    train.loc[train.State_Country == 'Rhode Island_US', 'Population'] = 1056161

    train.loc[train.State_Country == 'South Carolina_US', 'Population'] = 5210095

    train.loc[train.State_Country == 'South Dakota_US', 'Population'] = 903027

    train.loc[train.State_Country == 'Tennessee_US', 'Population'] = 6897576

    train.loc[train.State_Country == 'Texas_US', 'Population'] = 29472295

    train.loc[train.State_Country == 'Utah_US', 'Population'] = 3282115

    train.loc[train.State_Country == 'Vermont_US', 'Population'] = 628061

    train.loc[train.State_Country == 'Virgin Islands_US', 'Population'] = 104425

    train.loc[train.State_Country == 'Virginia_US', 'Population'] = 8626207

    train.loc[train.State_Country == 'Washington_US', 'Population'] = 7797095

    train.loc[train.State_Country == 'West Virginia_US', 'Population'] = 1778070

    train.loc[train.State_Country == 'Wisconsin_US', 'Population'] = 5851754

    train.loc[train.State_Country == 'Wyoming_US', 'Population'] = 567025

    

    return train



train = clear_population(train)

test = clear_population(test)
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
def make_simple(train, test):

    

    train = pd.read_csv('../input/covid19-global-forecasting-week-3/train.csv')

    test = pd.read_csv('../input/covid19-global-forecasting-week-3/test.csv')



    train['State_Country'] = [s + '_' + c if s == s else c for s,c in train[['Province_State', 'Country_Region']].values ]

    test['State_Country'] = [s + '_' + c if s == s else c for s,c in test[['Province_State', 'Country_Region']].values ]



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

test['New_Fatalities'] = (test['New_Fatalities'] * 0.6  + test['Fatalities'] * 0.4)



test['New_ConfirmedCases'] = test['ForecastId'].map(dict(zip(new_test_s['ForecastId'], new_test_s['ConfirmedCases'])))

test['New_ConfirmedCases'] = (test['New_ConfirmedCases'] * 0.6 + test['ConfirmedCases'] * 0.4 )
test['New_Fatalities'] = test['New_Fatalities'].clip(0, test['New_ConfirmedCases'] * 0.2)
submit = pd.read_csv('../input/covid19-global-forecasting-week-3/submission.csv')

submit['Fatalities'] = test['New_Fatalities'].astype('float')

submit['ConfirmedCases'] = test['New_ConfirmedCases'].astype('float')

submit.to_csv('submission.csv',index=False)
# bad_country = ['Malawi', 'Saint Pierre and Miquelon_France', 'South Sudan', 'Falkland Islands (Malvinas)_United Kingdom', 'Sao Tome and Principe', 'Bonaire, Sint Eustatius and Saba_Netherlands', 'Western Sahara']



# real_train = pd.read_csv('../input/covid19-global-forecasting-week-4/train.csv')

# real_train['State_Country'] = [s + '_' + c if s == s else c for s,c in real_train[['Province_State', 'Country_Region']].values ]



# new_test = test.merge(real_train[['State_Country', 'Date','ConfirmedCases', 'Fatalities']], on = ['State_Country', 'Date'])

# new_test = new_test[(new_test.Date >  '2020-04-08') & (~new_test.State_Country.isin(bad_country))]

# new_test = new_test.merge(new_test_s[['State_Country', 'Date','ConfirmedCases', 'Fatalities']], on = ['State_Country', 'Date'])

# # rmsle(new_test['ConfirmedCases_y'], new_test['ConfirmedCases_x'])