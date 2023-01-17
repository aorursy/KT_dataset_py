import numpy as np

import pandas as pd



import plotly.graph_objects as go

import plotly.express as px

import plotly.io as pio

pio.templates.default = "plotly_dark"

from plotly.subplots import make_subplots



from pathlib import Path

data_dir = Path('../input/')



import os

os.listdir(data_dir)
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_squared_error, mean_absolute_error
os.listdir('../input/covid19-global-forecasting-week-4/')
df = pd.read_csv('../input/covid19-global-forecasting-week-4/train.csv')

# df.rename(columns={'Country_Region' : 'country'}, inplace=True)

df
test = pd.read_csv('../input/covid19-global-forecasting-week-4/test.csv')

test[ test['Country_Region'] == 'India']
sample = pd.read_csv('../input/covid19-global-forecasting-week-4/submission.csv')

sample
test
df = pd.concat([df , test])

df
icu_df = pd.read_csv("../input/hospital-beds-by-country/API_SH.MED.BEDS.ZS_DS2_en_csv_v2_887506.csv")

icu_df['Country Name'] = icu_df['Country Name'].replace('United States', 'US')

icu_df['Country Name'] = icu_df['Country Name'].replace('Russian Federation', 'Russia')

icu_df['Country Name'] = icu_df['Country Name'].replace('Iran, Islamic Rep.', 'Iran')

icu_df['Country Name'] = icu_df['Country Name'].replace('Egypt, Arab Rep.', 'Egypt')

icu_df['Country Name'] = icu_df['Country Name'].replace('Venezuela, RB', 'Venezuela')

df['Country_Region'] = df['Country_Region'].replace('Czechia', 'Czech Republic')





# We wish to have the most recent values, thus we need to go through every year and extract the most recent one, if it exists.

icu_cleaned = pd.DataFrame()

icu_cleaned["Country_Region"] = icu_df["Country Name"]

icu_cleaned["icu"] = np.nan



for year in range(1960, 2020):

    year_df = icu_df[str(year)].dropna()

    icu_cleaned["icu"].loc[year_df.index] = year_df.values



df = pd.merge(df, icu_cleaned, on='Country_Region' , how = 'left')
df['CS'] = df['Country_Region'].astype(str) + df['Date'].astype(str) + df['Province_State'].astype(str)



df
df_temperature = pd.read_csv("../input/covid19-global-weather-data/temperature_dataframe.csv")

df_temperature
df_temperature.rename(columns={'country' : 'Country_Region'}, inplace=True)
df_temperature['Country_Region'] = df_temperature['Country_Region'].replace('USA', 'US')

df_temperature['Country_Region'] = df_temperature['Country_Region'].replace('UK', 'United Kingdom')

df_temperature = df_temperature[["Country_Region",  "date", "humidity", "sunHour", "tempC", "windspeedKmph"]].reset_index()

df_temperature.rename(columns={'province': 'state'}, inplace=True)

df_temperature["Date"] = pd.to_datetime(df_temperature['date'])

# df_temperature['state'] = df_temperature['state'].fillna('')

df_t = df_temperature.groupby('Country_Region').mean()



df_t
# df_temperature['CS'] = df_temperature['Country_Region'].astype(str) + df_temperature['date'].astype(str)

# df_temperature
df["Date"] = pd.to_datetime(df['Date'])

df1 = pd.merge(df , df_t, on=['Country_Region'],how = 'left' )

# df1.to_csv("countries_icu_temp.csv")
df1
df1['Province_State'] = df1['Province_State'].fillna(df1['Country_Region'])

cols = ['icu', 'humidity', 'sunHour', 'tempC',

       'windspeedKmph']

for col in cols : 

    df1[col] = df1[col].fillna(df1[col].mean())

# df1 = df1.set_index('CS')
# df1['Province_State'].unique()
n_start_death = 100

n_start_death1 = 1



# fatality_top_countires = top_country_df.sort_values('fatalities', ascending=False).iloc[:n_countries]['country'].values

# country_df['date'] = pd.to_datetime(country_df['date'])



## DAYS SINCE



df_list = []

for country in df1['Province_State'].unique():

    this_country_df = df1.query('Province_State == @country')

    start_date = this_country_df.query('ConfirmedCases > @n_start_death')['Date'].min()

    start_date1 = this_country_df.query('ConfirmedCases > @n_start_death1')['Date'].min()

    this_country_df['DConfirmed'] = this_country_df['Date'] - start_date

    this_country_df['DConfirmed1'] = this_country_df['Date'] - start_date1

#     this_country_df = this_country_df.query('Date >= @start_date')

#     this_country_df['date_since'] = this_country_df['Date'] - start_date

#     this_country_df['ConfirmedCases'] = np.log10(this_country_df['fatalities'] + 1)

#     this_country_df['fatalities_log1p'] -= this_country_df['fatalities_log1p'].values[0]

    df_list.append(this_country_df)



tmpdf = pd.concat(df_list)

tmpdf['DConfirmed'] = tmpdf['DConfirmed'] / pd.Timedelta('1 days')
# tmpdf[2050:2100]

tmpdf['DConfirmed'] = tmpdf['DConfirmed'].fillna(-50)

tmpdf['DConfirmed1'] = tmpdf['DConfirmed1'].fillna(-50)



df1 = tmpdf
## split into yr , month , date

df1['y'] , df1['m'] ,df1['d'] = df1['Date'].astype(str).str.split('-').str

df1
len(df1)
# train.info()
df1.columns
df1 = df1[['ConfirmedCases', 'Date', 'Fatalities',

       'Province_State', 'icu', 'humidity', 'sunHour', 'tempC',

       'windspeedKmph' , 'DConfirmed' , 'ForecastId' ,  'DConfirmed1', 'y', 'm', 'd']]

# df1 = df1[['ConfirmedCases', 'Country_Region', 'Date', 'Fatalities',

#         'Province_State','DConfirmed' , 'ForecastId' ,  'DConfirmed1', 'y', 'm', 'd']]

# df1['ConfirmedCases'] = np.log(df1['ConfirmedCases'] +1 )

# df1['Fatalities'] = np.log(df1['Fatalities'] + 1)

df1
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

df1['Province_State'] = le.fit_transform(df1['Province_State'])

df1
train = df1[df1['ForecastId'].isnull()]

# train = df1[df1['Date'] < '2020-03-26']

train.head()

testing = df1[~df1['ForecastId'].isnull()]

# len(testing)





train1 = train[train['Date'] < '2020-03-26']

val = train[train['Date'] >= '2020-03-26']

train_X = train1.drop(['ConfirmedCases' , 'Fatalities'] , axis = 1)

train_y = train1[['ConfirmedCases' , 'Fatalities']] 



val_X = val.drop(['ConfirmedCases' , 'Fatalities'] , axis = 1)

val_y = val[['ConfirmedCases' , 'Fatalities']] 





testX = testing.drop(['ConfirmedCases' , 'Fatalities'] , axis = 1)





categorical = ['Country_Region' , 'Province_State']



def column_index(df, query_cols):

    cols = df.columns.values

    sidx = np.argsort(cols)

    return sidx[np.searchsorted(cols, query_cols, sorter=sidx)]

categorical_features_indices = column_index(train_X, categorical)
len(testing)
len(sample)
testing.info()

#train.describe()

from catboost import CatBoostRegressor



# model = LGBMRegressor(num_leaves = 85,learning_rate =10**-1.89,n_estimators=100,min_sum_hessian_in_leaf=(10**-4.1),min_child_samples =2,subsample =0.97,subsample_freq=10,

#                    colsample_bytree = 0.68,reg_lambda=10**1.4,random_state=1234,n_jobs=4)

model =  CatBoostRegressor(iterations= 500,

#                              learning_rate=0.001,

                             depth=16,

                             eval_metric='RMSE',

                             random_seed = 42,

                             bagging_temperature = 0.2,

                             od_type='Iter',

                             metric_period = 50,

                                 task_type = "GPU",

                                 devices='0:1',



                             od_wait=100)



    

model.fit(train_X, train_y['ConfirmedCases'],

                 eval_set=(val_X, val_y['ConfirmedCases']),

#                   cat_features=categorical_features_indices,

                  use_best_model=True)



## Start model 

# By implementing a regression model which tries to use the country input variables to predict the most recent number of infections and deaths as target, we can extract the relative feature importance. This can be done pretty well with a Random Forest Regressor.



# sample
len(sample)
testX.info()



testX['ConfirmedCases'] = model.predict(testX) 

# sample['ConfirmedCases'] = np.exp(sample['ConfirmedCases'])

# sample
pd.set_option('display.max_columns', None)

pd.set_option('display.max_rows', 500)

testX
testX.info()
try :

    testX['Province_State'] = le.inverse_transform(testX['Province_State'])

except :

    x = 1
testX[testX['Province_State'] == 'India']
model1 =  CatBoostRegressor(iterations=500,

#                              learning_rate=0.001,

                             depth=16,

                             eval_metric='RMSE',

                             random_seed = 42,

                             bagging_temperature = 0.2,

                             od_type='Iter',

                             metric_period = 50,

                             task_type = "GPU",

                             devices='0:1',

                             

                             od_wait=100)



model1.fit(train_X, train_y['Fatalities'],

                 eval_set=(val_X, val_y['Fatalities']),

#                   cat_features=categorical_features_indices,

                 use_best_model=True)



# ## Start model 

# By implementing a regression model which tries to use the country input variables to predict the most recent number of infections and deaths as target, we can extract the relative feature importance. This can be done pretty well with a Random Forest Regressor.
sample



testX.info()

testX['Fatalities'] = model1.predict(testX)

# sample['Fatalities'] = np.exp(sample['Fatalities'])

# sample['Fatalities'] = 2**(model1.predict(testX) - 1)

sample = sample.set_index(['ForecastId'])
sample.to_csv('./submission.csv')
sample[:50]