import numpy as np

import pandas as pd

import xgboost as xgb

from xgboost import plot_importance, plot_tree

from sklearn.metrics import mean_squared_error, mean_absolute_error

from google.cloud import bigquery
train = pd.read_csv("../input/covid19-global-forecasting-week-1/train.csv")

test = pd.read_csv("../input/covid19-global-forecasting-week-1/test.csv")
test
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
from scipy.spatial.distance import cdist



weather_df['day_from_jan_first'] = (weather_df['da'].apply(int)

                                   + 31*(weather_df['mo']=='02') 

                                   + 60*(weather_df['mo']=='03')

                                   + 91*(weather_df['mo']=='04')  

                                   )



mo = train['Date'].apply(lambda x: x[5:7])

da = train['Date'].apply(lambda x: x[8:10])

train['day_from_jan_first'] = (da.apply(int)

                               + 31*(mo=='02') 

                               + 60*(mo=='03')

                               + 91*(mo=='04')  

                              )



C = []

for j in train.index:

    df = train.iloc[j:(j+1)]

    mat = cdist(df[['Lat','Long', 'day_from_jan_first']],

                weather_df[['lat','lon', 'day_from_jan_first']], 

                metric='euclidean')

    new_df = pd.DataFrame(mat, index=df.Id, columns=weather_df.index)

    arr = new_df.values

    new_close = np.where(arr == np.nanmin(arr, axis=1)[:,None],new_df.columns,False)

    L = [i[i.astype(bool)].tolist()[0] for i in new_close]

    C.append(L[0])

    

train['closest_station'] = C



train = train.set_index('closest_station').join(weather_df[['temp', 'min', 'max', 'stp', 'wdsp', 'prcp', 'fog']], ).reset_index().drop(['index'], axis=1)

train.sort_values(by=['Id'], inplace=True)

train.head()
from scipy.spatial.distance import cdist



weather_df['day_from_jan_first'] = (weather_df['da'].apply(int)

                                   + 31*(weather_df['mo']=='02') 

                                   + 60*(weather_df['mo']=='03')

                                   + 91*(weather_df['mo']=='04')  

                                   )



mo = test['Date'].apply(lambda x: x[5:7])

da = test['Date'].apply(lambda x: x[8:10])

test['day_from_jan_first'] = (da.apply(int)

                               + 31*(mo=='02') 

                               + 60*(mo=='03')

                               + 91*(mo=='04')  

                              )



C = []

for j in test.index:

    df = test.iloc[j:(j+1)]

    mat = cdist(df[['Lat','Long', 'day_from_jan_first']],

                weather_df[['lat','lon', 'day_from_jan_first']], 

                metric='euclidean')

    new_df = pd.DataFrame(mat, index=df.ForecastId, columns=weather_df.index)

    arr = new_df.values

    new_close = np.where(arr == np.nanmin(arr, axis=1)[:,None],new_df.columns,False)

    L = [i[i.astype(bool)].tolist()[0] for i in new_close]

    C.append(L[0])

    

test['closest_station'] = C



test = test.set_index('closest_station').join(weather_df[['temp', 'min', 'max', 'stp', 'wdsp', 'prcp', 'fog']], ).reset_index().drop(['index'], axis=1)

test.sort_values(by=['ForecastId'], inplace=True)

test.head()
train["wdsp"] = pd.to_numeric(train["wdsp"])

test["wdsp"] = pd.to_numeric(test["wdsp"])
train["fog"] = pd.to_numeric(train["fog"])

test["fog"] = pd.to_numeric(test["fog"])
X_train = train.drop(["Fatalities", "ConfirmedCases"], axis=1)
countries = X_train["Country/Region"]
countries.unique()
X_train = X_train.drop(["Id"], axis=1)

X_test = test.drop(["ForecastId"], axis=1)
X_train.dtypes
X_train['Date']= pd.to_datetime(X_train['Date']) 

X_test['Date']= pd.to_datetime(X_test['Date']) 
X_train = X_train.set_index(['Date'])

X_test = X_test.set_index(['Date'])
def create_time_features(df):

    """

    Creates time series features from datetime index

    """

    df['date'] = df.index

    df['hour'] = df['date'].dt.hour

    df['dayofweek'] = df['date'].dt.dayofweek

    df['quarter'] = df['date'].dt.quarter

    df['month'] = df['date'].dt.month

    df['year'] = df['date'].dt.year

    df['dayofyear'] = df['date'].dt.dayofyear

    df['dayofmonth'] = df['date'].dt.day

    df['weekofyear'] = df['date'].dt.weekofyear

    

    X = df[['hour','dayofweek','quarter','month','year',

           'dayofyear','dayofmonth','weekofyear']]

    return X
create_time_features(X_train)

create_time_features(X_test)
X_train
X_test
X_train.drop("date", axis=1, inplace=True)

X_test.drop("date", axis=1, inplace=True)
world_happiness_index = pd.read_csv("../input/world-bank-datasets/World_Happiness_Index.csv")
world_happiness_grouped = world_happiness_index.groupby('Country name').nth(-1)
world_happiness_grouped.drop("Year", axis=1, inplace=True)
X_train = pd.merge(left=X_train, right=world_happiness_grouped, how='left', left_on='Country/Region', right_on='Country name')

X_test = pd.merge(left=X_test, right=world_happiness_grouped, how='left', left_on='Country/Region', right_on='Country name')
X_train
malaria_world_health = pd.read_csv("../input/world-bank-datasets/Malaria_World_Health_Organization.csv")
X_train = pd.merge(left=X_train, right=malaria_world_health, how='left', left_on='Country/Region', right_on='Country')

X_test = pd.merge(left=X_test, right=malaria_world_health, how='left', left_on='Country/Region', right_on='Country')
X_train
X_train.drop("Country", axis=1, inplace=True)

X_test.drop("Country", axis=1, inplace=True)
human_development_index = pd.read_csv("../input/world-bank-datasets/Human_Development_Index.csv")
X_train = pd.merge(left=X_train, right=human_development_index, how='left', left_on='Country/Region', right_on='Country')

X_test = pd.merge(left=X_test, right=human_development_index, how='left', left_on='Country/Region', right_on='Country')
X_train
X_train.drop(["Country", "Gross national income (GNI) per capita 2018"], axis=1, inplace=True)

X_test.drop(["Country", "Gross national income (GNI) per capita 2018"], axis=1, inplace=True)
night_ranger_predictors = pd.read_csv("../input/covid19-demographic-predictors/covid19_by_country.csv")
#There is a duplicate for Georgia in this dataset from Night Ranger, causing merge issues so we will just drop the Georgia rows

night_ranger_predictors = night_ranger_predictors[night_ranger_predictors.Country != "Georgia"]
X_train = pd.merge(left=X_train, right=night_ranger_predictors, how='left', left_on='Country/Region', right_on='Country')

X_test = pd.merge(left=X_test, right=night_ranger_predictors, how='left', left_on='Country/Region', right_on='Country')
X_train
X_train.drop(["Country", "Restrictions","Quarantine", "Schools", "Total Infected", "Total Deaths"], axis=1, inplace=True)

X_test.drop(["Country", "Restrictions","Quarantine", "Schools", "Total Infected", "Total Deaths"], axis=1, inplace=True)
X_train.info(verbose=True)

X_test
X_train = pd.concat([X_train,pd.get_dummies(X_train['Province/State'], prefix='ps')],axis=1)

X_train.drop(['Province/State'],axis=1, inplace=True)

X_test = pd.concat([X_test,pd.get_dummies(X_test['Province/State'], prefix='ps')],axis=1)

X_test.drop(['Province/State'],axis=1, inplace=True)
X_train = pd.concat([X_train,pd.get_dummies(X_train['Country/Region'], prefix='cr')],axis=1)

X_train.drop(['Country/Region'],axis=1, inplace=True)

X_test = pd.concat([X_test,pd.get_dummies(X_test['Country/Region'], prefix='cr')],axis=1)

X_test.drop(['Country/Region'],axis=1, inplace=True)
y_train = train["Fatalities"]
y_train
X_train
reg = xgb.XGBRegressor(n_estimators=1000)
reg.fit(X_train, y_train, verbose=True)
plot = plot_importance(reg, height=0.9, max_num_features=20)
y_train = train.groupby(["Country/Region"]).Fatalities.pct_change(periods=1)
y_train = y_train.replace(np.nan, 0)
y_train = y_train.replace(np.inf, 0)
reg = xgb.XGBRegressor(n_estimators=1000)
reg.fit(X_train, y_train, verbose=True)
plot = plot_importance(reg, height=0.9, max_num_features=20)
y_train = train["ConfirmedCases"]
reg = xgb.XGBRegressor(n_estimators=1000)
reg.fit(X_train, y_train, verbose=True)
plot = plot_importance(reg, height=0.9, max_num_features=20)
y_train = train.groupby(["Country/Region"]).ConfirmedCases.pct_change(periods=1)
y_train = y_train.replace(np.nan, 0)
y_train = y_train.replace(np.inf, 0)
reg = xgb.XGBRegressor(n_estimators=1000)
reg.fit(X_train, y_train, verbose=True)
plot = plot_importance(reg, height=0.9, max_num_features=20)
y_train = train["ConfirmedCases"]

confirmed_reg = xgb.XGBRegressor(n_estimators=1000)

confirmed_reg.fit(X_train, y_train, verbose=True)

preds = confirmed_reg.predict(X_test)

preds = np.array(preds)

preds[preds < 0] = 0

preds = np.round(preds, 0)
preds = np.array(preds)
preds
submissionOrig = pd.read_csv("../input/covid19-global-forecasting-week-1/submission.csv")
submissionOrig["ConfirmedCases"]=pd.Series(preds)
submissionOrig
test = test.join(submissionOrig["ConfirmedCases"])

test["Difference"] = test.groupby(["Country/Region"])["ConfirmedCases"].apply(lambda x: x.shift(1) - x)

for index, row in test.iterrows():

    if index>0:

        if row["Difference"] < 0:

            test.at[index,"ConfirmedCases"] = test.iloc[index-1]["ConfirmedCases"]
submissionOrig["ConfirmedCases"] = test["ConfirmedCases"]

test.drop("ConfirmedCases", axis=1, inplace=True)
y_train = train["Fatalities"]

confirmed_reg = xgb.XGBRegressor(n_estimators=1000)

confirmed_reg.fit(X_train, y_train, verbose=True)

preds = confirmed_reg.predict(X_test)

preds = np.array(preds)

preds[preds < 0] = 0

preds = np.round(preds, 0)

submissionOrig["Fatalities"]=pd.Series(preds)
test = test.join(submissionOrig["Fatalities"])

test["Difference"] = test.groupby(["Country/Region"])["Fatalities"].apply(lambda x: x.shift(1) - x)

for index, row in test.iterrows():

    if index>0:

        if row["Difference"] < 0:

            test.at[index,"Fatalities"] = test.iloc[index-1]["Fatalities"]
submissionOrig["Fatalities"] = test["Fatalities"]

test.drop("Fatalities", axis=1, inplace=True)
submissionOrig
submissionOrig.to_csv('submission.csv',index=False)