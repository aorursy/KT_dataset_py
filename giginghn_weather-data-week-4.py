import numpy as np

import pandas as pd



import os

import json

from pathlib import Path



import matplotlib.pyplot as plt

from matplotlib import colors

import numpy as np



%matplotlib inline

import matplotlib.pyplot as plt

import seaborn as sns



from scipy.spatial.distance import cdist



for dirname, _, filenames in os.walk('/kaggle/input'):

    print(dirname)



from google.cloud import bigquery



from sklearn.experimental import enable_iterative_imputer

from sklearn.impute import IterativeImputer



#impute missing values

from sklearn.impute import KNNImputer

from sklearn.impute import SimpleImputer



from sklearn.experimental import enable_iterative_imputer

from sklearn.impute import IterativeImputer





data_path = Path('/kaggle/input/covid19-global-forecasting-week-1/')

train = pd.read_csv(data_path / 'train.csv')

test = pd.read_csv(data_path / 'test.csv')



data_path = Path('/kaggle/input/covid19-global-forecasting-week-2/')

train_2 = pd.read_csv(data_path / 'train.csv')

test_2 = pd.read_csv(data_path / 'test.csv')



data_path = Path('/kaggle/input/covid19-global-forecasting-week-4/')

train_4 = pd.read_csv(data_path / 'train.csv')

test_4 = pd.read_csv(data_path / 'test.csv')
x = len([nation for nation in train_4['Country_Region'].unique()])

print("There are {} different nations in the training dataset.".format(x))



s = train_4.groupby('Country_Region').ConfirmedCases.max()

nations_cases = [x for x in s.index if s[x]]

print("{} of these nations have confirmed cases of COVID-19.".format(len(nations_cases)))



s_4 = train_4.groupby('Country_Region').Fatalities.max()

nations_deaths = [x for x in s_4.index if s_4[x]]

print("In {} of these nations, people have died of COVID-19.".format(len(nations_deaths)))

%%time

client = bigquery.Client()

dataset_ref = client.dataset("noaa_gsod", project="bigquery-public-data")

dataset = client.get_dataset(dataset_ref)



tables = list(client.list_tables(dataset))



table_ref = dataset_ref.table("stations")

table = client.get_table(table_ref)

stations_df = client.list_rows(table).to_dataframe()



table_ref = dataset_ref.table("gsod2020")

table = client.get_table(table_ref)

twenty_twenty_df = client.list_rows(table).to_dataframe()



stations_df['STN'] = stations_df['usaf'] + '-' + stations_df['wban']

twenty_twenty_df['STN'] = twenty_twenty_df['stn'] + '-' + twenty_twenty_df['wban']



cols_1 = ['STN', 'mo', 'da', 'temp', 'min', 'max', 'stp', 'wdsp', 'prcp', 'fog']

cols_2 = ['STN', 'country', 'state', 'call', 'lat', 'lon', 'elev']

weather_df = twenty_twenty_df[cols_1].join(stations_df[cols_2].set_index('STN'), on='STN')



weather_df.tail(10)
weather_df['day_from_jan_first'] = (weather_df['da'].apply(int)

                                   + 31*(weather_df['mo']=='02') 

                                   + 60*(weather_df['mo']=='03')

                                   + 91*(weather_df['mo']=='04')  

                                   )
display(train_4.head())



train['country+province'] = train['Country/Region'].fillna('') + '-' + train['Province/State'].fillna('')

train_4['country+province'] = train_4['Country_Region'].fillna('') + '-' + train_4['Province_State'].fillna('')



df = train.groupby('country+province')[['Lat', 'Long']].mean()

df.loc['United Kingdom-'] = df.loc['United Kingdom-United Kingdom']

df.loc['Diamond Princess-'] = df.loc['Cruise Ship-Diamond Princess']

df.loc['Denmark-'] = df.loc['Denmark-Denmark']

df.loc['France-'] = df.loc['France-France']

df.loc['Gambia-'] = df.loc['Gambia, The-']

df.loc['Netherlands-'] = df.loc['Netherlands-Netherlands']

df.loc['Dominica-'] = (15.3, -61.383333)

df.loc['Angola-'] = (-8.830833, 13.245)

df.loc['Bahamas-'] = (25.066667, -77.333333)

df.loc['Belize-'] = (17.498611, -88.188611)

df.loc['Botswana-']=(-24.653257,25.906792)

df.loc['Burma-']=(16.871311,96.199379)

df.loc['Burundi-']=(-3.361260,29.347916)

df.loc['Cabo Verde-'] = (14.916667, -23.516667)

df.loc['Canada-Northwest Territories'] = (62.453972, -114.371788)

df.loc['Canada-Yukon'] = (64.000000, -135.000000)

df.loc['Chad-'] = (12.134722, 15.055833)

df.loc['Denmark-Greenland'] = (64.181389, -51.694167)

df.loc['El Salvador-'] = (13.698889, -89.191389)

df.loc['Eritrea-'] = (15.322778, 38.925)

df.loc['Fiji-'] = (-18.166667, 178.45)

df.loc['France-Martinique'] = (14.666667, -61)

df.loc['France-New Caledonia'] = (-22.2758, 166.458)

df.loc['France-Saint Pierre and Miquelon'] = (46.77914, -56.1773)

df.loc['Grenada-'] = (12.05, -61.75)

df.loc['Guinea-Bissau-'] = (11.85, -15.566667)

df.loc['Haiti-'] = (18.533333, -72.333333)

df.loc['Laos-'] = (17.966667, 102.6)

df.loc['Libya-'] = (32.887222, 13.191389)

df.loc['Madagascar-'] = (-18.933333, 47.516667)

df.loc['Mali-'] = (12.639167, -8.002778)

df.loc['Malawi-'] = (-15.786111, 35.005833)

df.loc['Mozambique-'] = (-25.966667, 32.583333)

df.loc['MS Zaandam-'] = (52.442039, 4.829199)

df.loc['Netherlands-Bonaire, Sint Eustatius and Saba'] = (12.1683718, -68.308183)

df.loc['Netherlands-Sint Maarten'] = (18.052778, -63.0425)

df.loc['Nicaragua-'] = (12.136389, -86.251389)

df.loc['Niger-'] = (13.511667, 2.125278)

df.loc['Papua New Guinea-'] = (-9.478889, 147.149444)

df.loc['Saint Kitts and Nevis-'] = (17.3, -62.733333)

df.loc['Sao Tome and Principe-'] = (0.255436, 6.602781)

df.loc['Sierra Leone-'] = (8.484444, -13.234444)

df.loc['South Sudan-'] = (4.859363, 31.571251)

df.loc['Syria-'] = (33.513056, 36.291944)

df.loc['Timor-Leste-'] = (-8.566667, 125.566667)

df.loc['Uganda-'] = (0.313611, 32.581111)

df.loc['Zimbabwe-'] = (-17.829167, 31.052222)

df.loc['United Kingdom-Bermuda'] = (32.293, -64.782)

df.loc['United Kingdom-Isle of Man'] = (54.145, -4.482)

df.loc['United Kingdom-Anguilla'] = (18.227230, -63.048988)

df.loc['United Kingdom-British Virgin Islands'] = (18.436539, -64.618103)

df.loc['United Kingdom-Falkland Islands (Malvinas)'] = (-51.563412, -51.563412)

df.loc['United Kingdom-Turks and Caicos Islands'] = (21.804132, 21.804132)

df.loc['West Bank and Gaza-'] = (31.9521618, 35.2331543)

df.loc['Western Sahara-'] = (27.154339, -13.199891)







train_4['Lat'] = train_4['country+province'].apply(lambda x: df.loc[x, 'Lat'])

train_4['Long'] = train_4['country+province'].apply(lambda x: df.loc[x, 'Long'])

mo = train_4['Date'].apply(lambda x: x[5:7])

da = train_4['Date'].apply(lambda x: x[8:10])

train_4['day_from_jan_first'] = (da.apply(int)

                               + 31*(mo=='02') 

                               + 60*(mo=='03')

                               + 91*(mo=='04')  

                              )



C = []

for j in train_4.index:

    df = train_4.iloc[j:(j+1)]

    mat = cdist(df[['Lat','Long', 'day_from_jan_first']],

                weather_df[['lat','lon', 'day_from_jan_first']], 

                metric='euclidean')

    new_df = pd.DataFrame(mat, index=df.Id, columns=weather_df.index)

    arr = new_df.values

    new_close = np.where(arr == np.nanmin(arr, axis=1)[:,None],new_df.columns,False)

    L = [i[i.astype(bool)].tolist()[0] for i in new_close]

    C.append(L[0])

    

train_4['closest_station'] = C



train_4= train_4.set_index('closest_station').join(weather_df[['temp', 'min', 'max', 'stp', 'wdsp', 'prcp', 'fog']], ).reset_index().drop(['index'], axis=1)

train_4.sort_values(by=['Id'], inplace=True)

train_4.index = train_4['Id'].apply(lambda x: x-1)

display(train_4.head())



#Export data

train_4.to_csv('training_data_with_weather_info_week_4.csv')
display(test_4.head())



test['country+province'] = test['Country/Region'].fillna('') + '-' + test['Province/State'].fillna('')

test_4['country+province'] = test_4['Country_Region'].fillna('') + '-' + test_4['Province_State'].fillna('')



df = test.groupby('country+province')[['Lat', 'Long']].mean()

df.loc['United Kingdom-'] = df.loc['United Kingdom-United Kingdom']

df.loc['Diamond Princess-'] = df.loc['Cruise Ship-Diamond Princess']

df.loc['Denmark-'] = df.loc['Denmark-Denmark']

df.loc['France-'] = df.loc['France-France']

df.loc['Gambia-'] = df.loc['Gambia, The-']

df.loc['Netherlands-'] = df.loc['Netherlands-Netherlands']

df.loc['Dominica-'] = (15.3, -61.383333)

df.loc['Angola-'] = (-8.830833, 13.245)

df.loc['Bahamas-'] = (25.066667, -77.333333)

df.loc['Belize-'] = (17.498611, -88.188611)

df.loc['Botswana-']=(-24.653257,25.906792)

df.loc['Burma-']=(16.871311,96.199379)

df.loc['Burundi-']=(-3.361260,29.347916)

df.loc['Cabo Verde-'] = (14.916667, -23.516667)

df.loc['Canada-Northwest Territories'] = (62.453972, -114.371788)

df.loc['Canada-Yukon'] = (64.000000, -135.000000)

df.loc['Chad-'] = (12.134722, 15.055833)

df.loc['Denmark-Greenland'] = (64.181389, -51.694167)

df.loc['El Salvador-'] = (13.698889, -89.191389)

df.loc['Eritrea-'] = (15.322778, 38.925)

df.loc['Fiji-'] = (-18.166667, 178.45)

df.loc['France-Martinique'] = (14.666667, -61)

df.loc['France-New Caledonia'] = (-22.2758, 166.458)

df.loc['France-Saint Pierre and Miquelon'] = (46.77914, -56.1773)

df.loc['Grenada-'] = (12.05, -61.75)

df.loc['Guinea-Bissau-'] = (11.85, -15.566667)

df.loc['Haiti-'] = (18.533333, -72.333333)

df.loc['Laos-'] = (17.966667, 102.6)

df.loc['Libya-'] = (32.887222, 13.191389)

df.loc['Madagascar-'] = (-18.933333, 47.516667)

df.loc['Mali-'] = (12.639167, -8.002778)

df.loc['Malawi-'] = (-15.786111, 35.005833)

df.loc['Mozambique-'] = (-25.966667, 32.583333)

df.loc['MS Zaandam-'] = (52.442039, 4.829199)

df.loc['Netherlands-Bonaire, Sint Eustatius and Saba'] = (12.1683718, -68.308183)

df.loc['Netherlands-Sint Maarten'] = (18.052778, -63.0425)

df.loc['Nicaragua-'] = (12.136389, -86.251389)

df.loc['Niger-'] = (13.511667, 2.125278)

df.loc['Papua New Guinea-'] = (-9.478889, 147.149444)

df.loc['Saint Kitts and Nevis-'] = (17.3, -62.733333)

df.loc['Sao Tome and Principe-'] = (0.255436, 6.602781)

df.loc['Sierra Leone-'] = (8.484444, -13.234444)

df.loc['South Sudan-'] = (4.859363, 31.571251)

df.loc['Syria-'] = (33.513056, 36.291944)

df.loc['Timor-Leste-'] = (-8.566667, 125.566667)

df.loc['Uganda-'] = (0.313611, 32.581111)

df.loc['Zimbabwe-'] = (-17.829167, 31.052222)

df.loc['United Kingdom-Bermuda'] = (32.293, -64.782)

df.loc['United Kingdom-Isle of Man'] = (54.145, -4.482)

df.loc['United Kingdom-Anguilla'] = (18.227230, -63.048988)

df.loc['United Kingdom-British Virgin Islands'] = (18.436539, -64.618103)

df.loc['United Kingdom-Falkland Islands (Malvinas)'] = (-51.563412, -51.563412)

df.loc['United Kingdom-Turks and Caicos Islands'] = (21.804132, 21.804132)

df.loc['West Bank and Gaza-'] = (31.9521618, 35.2331543)

df.loc['Western Sahara-'] = (27.154339, -13.199891)



test_4['Lat'] = test_4['country+province'].apply(lambda x: df.loc[x, 'Lat'])

test_4['Long'] = test_4['country+province'].apply(lambda x: df.loc[x, 'Long'])

mo = test_4['Date'].apply(lambda x: x[5:7])

da = test_4['Date'].apply(lambda x: x[8:10])

test_4['day_from_jan_first'] = (da.apply(int)

                               + 31*(mo=='02') 

                               + 60*(mo=='03')

                               + 91*(mo=='04')  

                              )



C = []

for j in test_4.index:

    df = test_4.iloc[j:(j+1)]

    mat = cdist(df[['Lat','Long', 'day_from_jan_first']],

                weather_df[['lat','lon', 'day_from_jan_first']], 

                metric='euclidean')

    new_df = pd.DataFrame(mat, index=df.ForecastId, columns=weather_df.index)

    arr = new_df.values

    new_close = np.where(arr == np.nanmin(arr, axis=1)[:,None],new_df.columns,False)

    L = [i[i.astype(bool)].tolist()[0] for i in new_close]

    C.append(L[0])

    

test_4['closest_station'] = C



test_4= test_4.set_index('closest_station').join(weather_df[['temp', 'min', 'max', 'stp', 'wdsp', 'prcp', 'fog']], ).reset_index().drop(['index'], axis=1)

test_4.sort_values(by=['ForecastId'], inplace=True)

test_4.index = test_4['ForecastId'].apply(lambda x: x-1)

display(test_4.head())



#Export data



test_4.to_csv('testing_data_with_weather_info_week_4.csv')