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

#import plotly.express as px

#import matplotlib.pyplot as plt

#import plotly.graph_objects as go

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

import numpy as np

from sklearn.metrics import mean_squared_log_error

from sklearn.tree import DecisionTreeRegressor

from google.cloud import bigquery

from scipy.spatial.distance import cdist

from sklearn.preprocessing import LabelEncoder
train_df=pd.read_csv("../input/covid19-global-forecasting-week-2/train.csv")

test_df=pd.read_csv("../input/covid19-global-forecasting-week-2/test.csv")
train_df.head()
train_df.info()
test_df.head()
print("Min train date: ",train_df["Date"].min())

print("Max train date: ",train_df["Date"].max())

print("Min test date: ",test_df["Date"].min())

print("Max test date: ",test_df["Date"].max())
train_df=train_df[train_df["Date"]<"2020-03-19"]
train_df.isnull().sum()
test_df.isnull().sum()
pop_info = pd.read_csv("../input/population-by-country-2020/population_by_country_2020.csv")
pop_info.rename(columns={'Density (P/Km²)': 'Density'}, inplace=True)
pop_info.columns
country_lookup=pop_info[["Country (or dependency)","Population (2020)","Density"]]
pd.DataFrame.from_dict(country_lookup)

train_df_pop=pd.merge(train_df, country_lookup, how='left', left_on='Country_Region', right_on='Country (or dependency)')
train_df_pop.info()
train_df_pop.loc[train_df_pop["Country_Region"]=="US", ["Population (2020)"]]=331002651 #United Sates

train_df_pop.loc[train_df_pop["Country_Region"]=="US", ["Density"]]=36

train_df_pop.loc[train_df_pop["Country_Region"]=="Korea, South", ["Population (2020)"]]=51269185 #South Korea

train_df_pop.loc[train_df_pop["Country_Region"]=="Korea, South", ["Density"]]=527

train_df_pop.loc[train_df_pop["Country_Region"]=="Czechia", ["Population (2020)"]]=10708981 #Czech Republic

train_df_pop.loc[train_df_pop["Country_Region"]=="Czechia", ["Density"]]=139

train_df_pop.loc[train_df_pop["Country_Region"]=="Taiwan*", ["Population (2020)"]]=23816775 #Taiwan

train_df_pop.loc[train_df_pop["Country_Region"]=="Taiwan*", ["Density"]]=673

train_df_pop.loc[train_df_pop["Country_Region"]=="Congo (Kinshasa)", ["Population (2020)"]]=89561403 #DR Congo

train_df_pop.loc[train_df_pop["Country_Region"]=="Congo (Kinshasa)", ["Density"]]=40

train_df_pop.loc[train_df_pop["Country_Region"]=="Congo (Brazzaville)", ["Population (2020)"]]=5518087 #Congo

train_df_pop.loc[train_df_pop["Country_Region"]=="Congo (Brazzaville)", ["Density"]]=16

train_df_pop.loc[train_df_pop["Country_Region"]=="Cote d'Ivoire", ["Population (2020)"]]=26378274 #CÃ´te d'Ivoire

train_df_pop.loc[train_df_pop["Country_Region"]=="Cote d'Ivoire", ["Density"]]=83

train_df_pop.loc[train_df_pop["Country_Region"]=="Saint Kitts and Nevis", ["Population (2020)"]]=53199 #Saint Kitts & Nevis

train_df_pop.loc[train_df_pop["Country_Region"]=="Saint Kitts and Nevis", ["Density"]]=205

train_df_pop.loc[train_df_pop["Country_Region"]=="Saint Vincent and the Grenadines", ["Population (2020)"]]=110940 #St. Vincent & Grenadines

train_df_pop.loc[train_df_pop["Country_Region"]=="Saint Vincent and the Grenadines", ["Density"]]=284

train_df_pop.loc[train_df_pop["Country_Region"]=="Diamond Princess", ["Population (2020)"]]=3770 #Population and density are same since it is a cruise ship

train_df_pop.loc[train_df_pop["Country_Region"]=="Diamond Princess", ["Density"]]=3770
test_df_pop=pd.merge(test_df, country_lookup, how='left', left_on='Country_Region', right_on='Country (or dependency)')
test_df_pop.info()
test_df_pop.loc[test_df_pop["Country_Region"]=="US", ["Population (2020)"]]=331002651 #United Sates

test_df_pop.loc[test_df_pop["Country_Region"]=="US", ["Density"]]=36

test_df_pop.loc[test_df_pop["Country_Region"]=="Korea, South", ["Population (2020)"]]=51269185 #South Korea

test_df_pop.loc[test_df_pop["Country_Region"]=="Korea, South", ["Density"]]=527

test_df_pop.loc[test_df_pop["Country_Region"]=="Czechia", ["Population (2020)"]]=10708981 #Czech Republic

test_df_pop.loc[test_df_pop["Country_Region"]=="Czechia", ["Density"]]=139

test_df_pop.loc[test_df_pop["Country_Region"]=="Taiwan*", ["Population (2020)"]]=23816775 #Taiwan

test_df_pop.loc[test_df_pop["Country_Region"]=="Taiwan*", ["Density"]]=673

test_df_pop.loc[test_df_pop["Country_Region"]=="Congo (Kinshasa)", ["Population (2020)"]]=89561403 #DR Congo

test_df_pop.loc[test_df_pop["Country_Region"]=="Congo (Kinshasa)", ["Density"]]=40

test_df_pop.loc[test_df_pop["Country_Region"]=="Congo (Brazzaville)", ["Population (2020)"]]=5518087 #Congo

test_df_pop.loc[test_df_pop["Country_Region"]=="Congo (Brazzaville)", ["Density"]]=16

test_df_pop.loc[test_df_pop["Country_Region"]=="Cote d'Ivoire", ["Population (2020)"]]=26378274 #CÃ´te d'Ivoire

test_df_pop.loc[test_df_pop["Country_Region"]=="Cote d'Ivoire", ["Density"]]=83

test_df_pop.loc[test_df_pop["Country_Region"]=="Saint Kitts and Nevis", ["Population (2020)"]]=53199 #Saint Kitts & Nevis

test_df_pop.loc[test_df_pop["Country_Region"]=="Saint Kitts and Nevis", ["Density"]]=205

test_df_pop.loc[test_df_pop["Country_Region"]=="Saint Vincent and the Grenadines", ["Population (2020)"]]=110940 #St. Vincent & Grenadines

test_df_pop.loc[test_df_pop["Country_Region"]=="Saint Vincent and the Grenadines", ["Density"]]=284

test_df_pop.loc[test_df_pop["Country_Region"]=="Diamond Princess", ["Population (2020)"]]=3770 #Population and density are same since it is a cruise ship

test_df_pop.loc[test_df_pop["Country_Region"]=="Diamond Princess", ["Density"]]=3770
train_df_pop.drop("Country (or dependency)", axis=1, inplace=True)
test_df_pop.drop("Country (or dependency)", axis=1, inplace=True)
%load_ext google.cloud.bigquery
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



weather_df["da"]=weather_df["da"].astype(int)
weather_df['day_from_jan_first'] = weather_df["da"] + 31*(weather_df["mo"]=='02') + 60*(weather_df["mo"]=='03') + 91*(weather_df["mo"]=='04')  

                              
train_wk1=pd.read_csv("../input/training-dataset-from-covid-19-week-1-forecasting/train-3.csv")

train_wk1['country+province'] = train_wk1['Country/Region'].fillna('') + '-' + train_wk1['Province/State'].fillna('')

train_df_pop['country+province'] = train_df_pop['Country_Region'].fillna('') + '-' + train_df_pop['Province_State'].fillna('')

test_df_pop['country+province'] = test_df_pop['Country_Region'].fillna('') + '-' + test_df_pop['Province_State'].fillna('')
df = train_wk1.groupby('country+province')[['Lat', 'Long']].mean()
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

df.loc['Cabo Verde-'] = (14.916667, -23.516667)

df.loc['Chad-'] = (12.134722, 15.055833)

df.loc['Denmark-Greenland'] = (64.181389, -51.694167)

df.loc['El Salvador-'] = (13.698889, -89.191389)

df.loc['Eritrea-'] = (15.322778, 38.925)

df.loc['Fiji-'] = (-18.166667, 178.45)

df.loc['France-Martinique'] = (14.666667, -61)

df.loc['France-New Caledonia'] = (-22.2758, 166.458)

df.loc['Grenada-'] = (12.05, -61.75)

df.loc['Guinea-Bissau-'] = (11.85, -15.566667)

df.loc['Haiti-'] = (18.533333, -72.333333)

df.loc['Laos-'] = (17.966667, 102.6)

df.loc['Libya-'] = (32.887222, 13.191389)

df.loc['Madagascar-'] = (-18.933333, 47.516667)

df.loc['Mali-'] = (12.639167, -8.002778)

df.loc['Mozambique-'] = (-25.966667, 32.583333)

df.loc['Netherlands-Sint Maarten'] = (18.052778, -63.0425)

df.loc['Nicaragua-'] = (12.136389, -86.251389)

df.loc['Niger-'] = (13.511667, 2.125278)

df.loc['Papua New Guinea-'] = (-9.478889, 147.149444)

df.loc['Saint Kitts and Nevis-'] = (17.3, -62.733333)

df.loc['Syria-'] = (33.513056, 36.291944)

df.loc['Timor-Leste-'] = (-8.566667, 125.566667)

df.loc['Uganda-'] = (0.313611, 32.581111)

df.loc['Zimbabwe-'] = (-17.829167, 31.052222)

df.loc['United Kingdom-Bermuda'] = (32.293, -64.782)

df.loc['United Kingdom-Isle of Man'] = (54.145, -4.482)
train_df_pop['Lat'] = train_df_pop['country+province'].apply(lambda x: df.loc[x, 'Lat'])

train_df_pop['Long'] = train_df_pop['country+province'].apply(lambda x: df.loc[x, 'Long'])

test_df_pop['Lat'] = test_df_pop['country+province'].apply(lambda x: df.loc[x, 'Lat'])

test_df_pop['Long'] = test_df_pop['country+province'].apply(lambda x: df.loc[x, 'Long'])

mo = train_df_pop['Date'].apply(lambda x: x[5:7])

da = train_df_pop['Date'].apply(lambda x: x[8:10])

mo_test = test_df_pop['Date'].apply(lambda x: x[5:7])

da_test = test_df_pop['Date'].apply(lambda x: x[8:10])

train_df_pop['day_from_jan_first'] = (da.apply(int)

                               + 31*(mo=='02') 

                               + 60*(mo=='03')

                               + 91*(mo=='04')  

                              )

test_df_pop['day_from_jan_first'] = (da_test.apply(int)

                               + 31*(mo_test=='02') 

                               + 60*(mo_test=='03')

                               + 91*(mo_test=='04')  

                              )
C = []

for j in train_df_pop.index:

    df_train = train_df_pop.iloc[j:(j+1)]

    mat = cdist(df_train[['Lat','Long', 'day_from_jan_first']],

                weather_df[['lat','lon', 'day_from_jan_first']], 

                metric='euclidean')

    new_df = pd.DataFrame(mat, index=df_train.Id, columns=weather_df.index)

    arr = new_df.values

    new_close = np.where(arr == np.nanmin(arr, axis=1)[:,None],new_df.columns,False)

    L = [i[i.astype(bool)].tolist()[0] for i in new_close]

    C.append(L[0])

    

train_df_pop['closest_station'] = C

train_df_pop= train_df_pop.set_index('closest_station').join(weather_df[['temp', 'min', 'max', 'stp', 'wdsp', 'prcp', 'fog']], ).reset_index().drop(['index'], axis=1)

train_df_pop.sort_values(by=['Id'], inplace=True)

train_df_pop.index = train_df_pop['Id'].apply(lambda x: x-1)
D=[]



for j in test_df_pop.index:

    df_test = test_df_pop.iloc[j:(j+1)]

    mat_test = cdist(df_test[['Lat','Long', 'day_from_jan_first']],

                weather_df[['lat','lon', 'day_from_jan_first']], 

                metric='euclidean')

    new_df_test = pd.DataFrame(mat_test, index=df_test.ForecastId, columns=weather_df.index)

    arr_test = new_df_test.values

    new_close_test = np.where(arr_test == np.nanmin(arr_test, axis=1)[:,None],new_df_test.columns,False)

    L_test = [i[i.astype(bool)].tolist()[0] for i in new_close_test]

    D.append(L_test[0])

    

test_df_pop['closest_station'] = D

test_df_pop= test_df_pop.set_index('closest_station').join(weather_df[['temp', 'min', 'max', 'stp', 'wdsp', 'prcp', 'fog']], ).reset_index().drop(['index'], axis=1)

test_df_pop.sort_values(by=['ForecastId'], inplace=True)

test_df_pop.index = test_df_pop['ForecastId'].apply(lambda x: x-1)
train_df_pop.reset_index(drop=True, inplace=True)
train_df_pop["Id"]=train_df_pop.index
train_df_pop.isnull().sum()
labelencoder = LabelEncoder()
train_df_pop['Country_Region_Types'] = labelencoder.fit_transform(train_df_pop['Country_Region'])
test_df_pop['Country_Region_Types'] = labelencoder.fit_transform(test_df_pop['Country_Region'])
train_df_pop.head()
test_df_pop.head()
train_df_pop.rename(columns={"Population (2020)":"Population"}, inplace=True)
test_df_pop.rename(columns={"Population (2020)":"Population"}, inplace=True)
train_df_pop.head()
test_df_pop.head()
X_dataset=train_df_pop.copy()
X_dataset=X_dataset[["Date","Population","Density","day_from_jan_first","temp","min","max","stp","wdsp","prcp","fog","Country_Region_Types","Lat","Long"]]
X_dataset["Date"] = X_dataset["Date"].apply(lambda x:x.replace("-",""))

X_dataset["Date"]  = X_dataset["Date"].astype(int)
X_dataset.info()
X_dataset.head()
X_dataset["fog"]  = X_dataset["fog"].astype(int)

X_dataset["wdsp"]  = X_dataset["wdsp"].astype(float)
X_dataset.info()
X_dataset["Date"].max()
y1=train_df_pop[train_df_pop["Date"]<"2020-03-19"]["ConfirmedCases"]

y2=train_df_pop[train_df_pop["Date"]<"2020-03-19"]["Fatalities"]
#Confirmed Cases

X_train_confirmed, X_test_confirmed, y_train_confirmed, y_test_confirmed = train_test_split(X_dataset, y1, test_size = .20, random_state = 42)
dt1=DecisionTreeRegressor(criterion="friedman_mse",max_depth=20,random_state=42)
dt1.fit(X_train_confirmed, y_train_confirmed)
y_pred_dt_confirmed=dt1.predict(X_test_confirmed)
np.sqrt(mean_squared_log_error( y_test_confirmed, y_pred_dt_confirmed ))
#Fatalities

X_train_fatal, X_test_fatal, y_train_fatal, y_test_fatal = train_test_split(X_dataset, y2, test_size = .20, random_state = 42)
dt2=DecisionTreeRegressor(criterion="friedman_mse",max_depth=20,random_state=42)
dt2.fit(X_train_fatal, y_train_fatal)
y_pred_dt_fatal=dt2.predict(X_test_fatal)
np.sqrt(mean_squared_log_error( y_test_fatal, y_pred_dt_fatal ))
test_data=test_df_pop[["Date","Population","Density","day_from_jan_first","temp","min","max","stp","wdsp","prcp","fog","Country_Region_Types","Lat","Long"]]
test_data.head()
test_data["Date"] = test_data["Date"].apply(lambda x:x.replace("-",""))

test_data["Date"]  = test_data["Date"].astype(int)
y_confirmed=dt1.predict(test_data)
y_fatal=dt2.predict(test_data)
submission=pd.DataFrame({'ForecastId': test_df["ForecastId"], 'ConfirmedCases': y_confirmed, 'Fatalities': y_fatal})
submission["ConfirmedCases"]=submission["ConfirmedCases"].astype(int)

submission["Fatalities"]=submission["Fatalities"].astype(int)
submission.head()
submission.to_csv('submission.csv', index=False)