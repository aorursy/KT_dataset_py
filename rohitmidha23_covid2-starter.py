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
import matplotlib.pyplot as plt

from matplotlib import colors



%matplotlib inline

import matplotlib.pyplot as plt

import seaborn as sns



from scipy.spatial.distance import cdist

from pathlib import Path

for dirname, _, filenames in os.walk('/kaggle/input'):

    print(dirname)



from google.cloud import bigquery

from datetime import datetime





data_path = Path('/kaggle/input/covid19-global-forecasting-week-1/')

train = pd.read_csv(data_path / 'train.csv')

test = pd.read_csv(data_path / 'test.csv')



data_path = Path('/kaggle/input/covid19-global-forecasting-week-2/')

train_2 = pd.read_csv(data_path / 'train.csv')

test_2 = pd.read_csv(data_path / 'test.csv')

# %%time

# client = bigquery.Client()

# dataset_ref = client.dataset("noaa_gsod", project="bigquery-public-data")

# dataset = client.get_dataset(dataset_ref)



# tables = list(client.list_tables(dataset))



# table_ref = dataset_ref.table("stations")

# table = client.get_table(table_ref)

# stations_df = client.list_rows(table).to_dataframe()



# table_ref = dataset_ref.table("gsod2020")

# table = client.get_table(table_ref)

# twenty_twenty_df = client.list_rows(table).to_dataframe()



# stations_df['STN'] = stations_df['usaf'] + '-' + stations_df['wban']

# twenty_twenty_df['STN'] = twenty_twenty_df['stn'] + '-' + twenty_twenty_df['wban']



# cols_1 = ['STN', 'mo', 'da', 'temp', 'min', 'max', 'stp', 'wdsp', 'prcp', 'fog']

# cols_2 = ['STN', 'country', 'state', 'call', 'lat', 'lon', 'elev']

# weather_df = twenty_twenty_df[cols_1].join(stations_df[cols_2].set_index('STN'), on='STN')



# weather_df.tail(10)
# weather_df['day_from_jan_first'] = (weather_df['da'].apply(int)

#                                    + 31*(weather_df['mo']=='02') 

#                                    + 60*(weather_df['mo']=='03')

#                                    + 91*(weather_df['mo']=='04')  

#                                    )



# mo = train['Date'].apply(lambda x: x[5:7])

# da = train['Date'].apply(lambda x: x[8:10])

# train['day_from_jan_first'] = (da.apply(int)

#                                + 31*(mo=='02') 

#                                + 60*(mo=='03')

#                                + 91*(mo=='04')  

#                               )



# C = []

# for j in train.index:

#     df = train.iloc[j:(j+1)]

#     mat = cdist(df[['Lat','Long', 'day_from_jan_first']],

#                 weather_df[['lat','lon', 'day_from_jan_first']], 

#                 metric='euclidean')

#     new_df = pd.DataFrame(mat, index=df.Id, columns=weather_df.index)

#     arr = new_df.values

#     new_close = np.where(arr == np.nanmin(arr, axis=1)[:,None],new_df.columns,False)

#     L = [i[i.astype(bool)].tolist()[0] for i in new_close]

#     C.append(L[0])

    

# train['closest_station'] = C



# train = train.set_index('closest_station').join(weather_df[['temp', 'min', 'max', 'stp', 'wdsp', 'prcp', 'fog']], ).reset_index().drop(['index'], axis=1)

# train.sort_values(by=['Id'], inplace=True)

# #train = train.set_index('Id')

# train.index = train['Id'].apply(lambda x: x-1)

# train.head()
# train.to_csv('training_data_with_weather_info_week_1.csv')
train['country+province'] = train['Country/Region'].fillna('') + '-' + train['Province/State'].fillna('')

train_2['country+province'] = train_2['Country_Region'].fillna('') + '-' + train_2['Province_State'].fillna('')



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



train_2['Lat'] = train_2['country+province'].apply(lambda x: df.loc[x, 'Lat'])

train_2['Long'] = train_2['country+province'].apply(lambda x: df.loc[x, 'Long'])

train_2.head()
train_2.to_csv("train2_latlong.csv",index=False)
test_2['country+province'] = test_2['Country_Region'].fillna('') + '-' + test_2['Province_State'].fillna('')



test_2['Lat'] = test_2['country+province'].apply(lambda x: df.loc[x, 'Lat'])

test_2['Long'] = test_2['country+province'].apply(lambda x: df.loc[x, 'Long'])

test_2.head()
test_2.to_csv("test2_latlong.csv",index=False)
# mo = train_2['Date'].apply(lambda x: x[5:7])

# da = train_2['Date'].apply(lambda x: x[8:10])

# train_2['day_from_jan_first'] = (da.apply(int)

#                                + 31*(mo=='02') 

#                                + 60*(mo=='03')

#                                + 91*(mo=='04')  

#                               )



# C = []

# for j in train_2.index:

#     df = train_2.iloc[j:(j+1)]

#     mat = cdist(df[['Lat','Long', 'day_from_jan_first']],

#                 weather_df[['lat','lon', 'day_from_jan_first']], 

#                 metric='euclidean')

#     new_df = pd.DataFrame(mat, index=df.Id, columns=weather_df.index)

#     arr = new_df.values

#     new_close = np.where(arr == np.nanmin(arr, axis=1)[:,None],new_df.columns,False)

#     L = [i[i.astype(bool)].tolist()[0] for i in new_close]

#     C.append(L[0])

    

# train_2['closest_station'] = C



# train_2= train_2.set_index('closest_station').join(weather_df[['temp', 'min', 'max', 'stp', 'wdsp', 'prcp', 'fog']], ).reset_index().drop(['index'], axis=1)

# train_2.sort_values(by=['Id'], inplace=True)

# #train_2 = train_2.set_index('Id')

# train_2.index = train_2['Id'].apply(lambda x: x-1)

# display(train_2.head())



# train_2.to_csv('training_data_with_weather_info_week_2.csv')
import lightgbm as lgb

from fastai.tabular import *

from sklearn import preprocessing

import datetime
train = pd.read_csv("train2_latlong.csv")

test = pd.read_csv("test2_latlong.csv")

sub = pd.read_csv("../input/covid19-global-forecasting-week-2/submission.csv")
train.head()
print(train.Date.min())

print(train.Date.max())

print(test.Date.min())

print(test.Date.max())
train = train.append(test[test['Date']>'2020-03-26'])
train.head()
train.shape
def rmsle (y_true, y_pred):

    return np.sqrt(np.mean(np.power(np.log1p(y_pred) - np.log1p(y_true), 2)))

def mape (y_true, y_pred):

    return np.mean(np.abs(y_pred -y_true)*100/(y_true+1))
train['Date'] = pd.to_datetime(train['Date'], format='%Y-%m-%d')
train['day_dist'] = train['Date']-train['Date'].min()

train['day_dist'] = train['day_dist'].dt.days

# test['day_dist'] = test['Date']-test['Date'].min()

# test['day_dist'] = test['day_dist'].dt.days

cat_cols = train.dtypes[train.dtypes=='object'].keys()

cat_cols
for cat_col in cat_cols:

    train[cat_col].fillna('no_value', inplace = True)

    

train['place'] = train['Province_State']+'_'+train['Country_Region']

for cat_col in ['place']:

    le = preprocessing.LabelEncoder()

    le.fit(train[cat_col])

    train[cat_col]=le.transform(train[cat_col])
train.keys()

train["Date"].max()
test["Date"].min()
# train["Date1"] = train["Date"]

# add_datepart(train, field_name="Date1")
val = train[(train['Date']>='2020-03-19')&(train['Id'].isnull()==False)]
y_ft = train["Fatalities"]

y_val_ft = val["Fatalities"]

y_cc = train["ConfirmedCases"]

y_val_cc = val["ConfirmedCases"]
params = {

    "objective": "regression",

    "boosting": 'gbdt',

    "num_leaves": 1280,

    "learning_rate": 0.05,

    "feature_fraction": 0.9,

    "reg_lambda": 2,

    "metric": "rmse",

    'min_data_in_leaf':20

}
dates = test['Date'].unique()

dates
dates = dates[dates>'2020-03-26']

len(dates)
drop_cols = ['Id', 'ConfirmedCases', 'Fatalities','day_dist', 'Province_State', 'Country_Region', 'Date', 'country+province']
places = train["country+province"].unique()

len(places)
# place="India-"

# initdate = min(train.loc[(train["country+province"]==place) & (train["ConfirmedCases"]>0)]["Date"])

# initdate
# train.loc[(train["country+province"]==place) & (train["ConfirmedCases"]>0)]
places_dict={}

for place in places:

    initdate = min(train.loc[(train["country+province"]==place) & (train["ConfirmedCases"]>0)]["Date"])

    places_dict[place] = initdate.date()

    

places_dict
days_from_first_case = []

for i in range(len(train)):

    row = train.iloc[i]

    row = row.to_dict()

    initdate = places_dict[row["country+province"]]

    currdate = row["Date"].date()

    days = (currdate - initdate).days

    days_from_first_case.append(max(0,days))
train['Days_from_first_case'] = days_from_first_case
train.head(50)
train.to_csv("finaltrain.csv",index=False)
i = 0

fold_n = 0

for date in dates:

    print("Date: ", date)

    fold_n = fold_n +1 

    i = i+1

    if i==1:

        nrounds = 200

    else:

        nrounds = 100

#     print(i)

#     print(nrounds)

    

    train['shift_1_cc'] = train.groupby(['place'])['ConfirmedCases'].shift(i)

    train['shift_2_cc'] = train.groupby(['place'])['ConfirmedCases'].shift(i+1)

    train['shift_3_cc'] = train.groupby(['place'])['ConfirmedCases'].shift(i+2)

    train['shift_4_cc'] = train.groupby(['place'])['ConfirmedCases'].shift(i+3)

    train['shift_5_cc'] = train.groupby(['place'])['ConfirmedCases'].shift(i+4)

        

    val2 = train[train['Date']==date]

    train2 = train[(train['Date']<date)]

    y_cc = train2["ConfirmedCases"]

#     print(val2.head())

#     y_val_cc = val2["ConfirmedCases"]

    

    train2.drop(drop_cols, axis=1, inplace=True)

    val2.drop(drop_cols, axis=1, inplace=True)

    

#     np.log1p(y)

#     feature_importances = pd.DataFrame()

#     feature_importances['feature'] = train.keys()

    

    #score = 0       

    dtrain = lgb.Dataset(train2, label=y_cc)

    dvalid = lgb.Dataset(val2, label=y_val_cc)



    model = lgb.train(params, dtrain, nrounds, 

#                             valid_sets = [dtrain, dvalid],

                            categorical_feature = ['place'], #'Province/State', 'Country/Region'

                            verbose_eval=False)#, early_stopping_rounds=50)



    y_pred = model.predict(val2,num_iteration=nrounds)  #model.best_iteration

#     y_pred = np.expm1( y_pred)

#     vcheck.loc[vcheck['Date']==date,'cc_predict'] = y_pred

    test.loc[test['Date']==date,'ConfirmedCases'] = y_pred

    train.loc[train['Date']==date,'ConfirmedCases'] = y_pred

#     y_oof[valid_index] = y_pred



#     rmsle_score = rmsle(y_val_cc, y_pred)

#     mape_score = mape(y_val_cc, y_pred)

#     score += rmsle_score

#     print (f'fold: {date}, rmsle: {rmsle_score:.5f}' )

#     print (f'fold: {date}, mape: {mape_score:.5f}' )
i = 0

fold_n = 0

for date in dates:

    print(date)

    fold_n = fold_n +1 

    i = i+1

    if i==1:

        nrounds = 200

    else:

        nrounds = 100

#     print(i)

#     print(nrounds)

    

    train['shift_1_cc'] = train.groupby(['place'])['Fatalities'].shift(i)

    train['shift_2_cc'] = train.groupby(['place'])['Fatalities'].shift(i+1)

    train['shift_3_cc'] = train.groupby(['place'])['Fatalities'].shift(i+2)

    train['shift_4_cc'] = train.groupby(['place'])['Fatalities'].shift(i+3)

    train['shift_5_cc'] = train.groupby(['place'])['Fatalities'].shift(i+4)

        

    val2 = train[train['Date']==date]

    train2 = train[(train['Date']<date)]

    y_ft = train2["Fatalities"]

    #y_val_cc = val2["ConfirmedCases"]

    

    train2.drop(drop_cols, axis=1, inplace=True)

    val2.drop(drop_cols, axis=1, inplace=True)

    

#     np.log1p(y)

#     feature_importances = pd.DataFrame()

#     feature_importances['feature'] = train.keys()



    dtrain = lgb.Dataset(train2, label=y_ft)

    dvalid = lgb.Dataset(val2, label=y_val_ft)



    model = lgb.train(params, dtrain, nrounds, 

#                             valid_sets = [dtrain, dvalid],

                            categorical_feature = ['place'], #'Province/State', 'Country/Region'

                            verbose_eval=False)#, early_stopping_rounds=50)



    y_pred = model.predict(val2,num_iteration=nrounds)  #model.best_iteration

#     y_pred = np.expm1( y_pred)

#     vcheck.loc[vcheck['Date']==date,'cc_predict'] = y_pred

    test.loc[test['Date']==date,'Fatalities'] = y_pred

    train.loc[train['Date']==date,'Fatalities'] = y_pred

#     y_oof[valid_index] = y_pred



#     rmsle_score = rmsle(y_val_cc, y_pred)

#     mape_score = mape(y_val_cc, y_pred)

#     score += rmsle_score

#     print (f'fold: {date}, rmsle: {rmsle_score:.5f}' )

#     print (f'fold: {date}, mape: {mape_score:.5f}' )

test[test['Country_Region']=='India']
train.head()
train_sub = pd.read_csv("../input/covid19-global-forecasting-week-2/train.csv")
train_sub.head()
test = pd.merge(test,train_sub[['Province_State','Country_Region','Date','ConfirmedCases','Fatalities']], on=['Province_State','Country_Region','Date'], how='left')
test.shape
test.head()
test.loc[test['ConfirmedCases_x'].isnull()==True, 'ConfirmedCases_x'] = test.loc[test['ConfirmedCases_x'].isnull()==True, 'ConfirmedCases_y']
test.head()
test.loc[test['Fatalities_x'].isnull()==True, 'Fatalities_x'] = test.loc[test['Fatalities_x'].isnull()==True, 'Fatalities_y']
test.head()
last_amount = test.loc[(test['Country_Region']=='Italy')&(test['Date']=='2020-03-26'),'ConfirmedCases_x']
last_fat = test.loc[(test['Country_Region']=='Italy')&(test['Date']=='2020-03-24'),'Fatalities_x']
last_amount, last_fat
i, k = 0, 35

for date in dates:

    k = k-1

    i = i + 1

    test.loc[(test['Country_Region']=='Italy')&(test['Date']==date),'ConfirmedCases_x'] =  last_amount.values[0]+i*(5000-(100*i))

    test.loc[(test['Country_Region']=='Italy')&(test['Date']==date),'Fatalities_x'] =  last_fat.values[0]+i*(800-(10*i))
test.loc[(test['Country_Region']=='Italy')]
sub = test[['ForecastId', 'ConfirmedCases_x','Fatalities_x']]

sub.columns = ['ForecastId', 'ConfirmedCases', 'Fatalities']
sub.head()
sub.loc[sub['ConfirmedCases']<0, 'ConfirmedCases'] = 0
sub.loc[sub['Fatalities']<0, 'Fatalities'] = 0
sub['Fatalities'].describe()
sub.to_csv('submission.csv',index=False)