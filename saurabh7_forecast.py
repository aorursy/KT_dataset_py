import pandas as pd

import matplotlib.pyplot as plt

from wordcloud import WordCloud





import matplotlib.pyplot as plt

# import seaborn as sns

import plotly.express as px

import plotly.graph_objects as go

from wordcloud import WordCloud

from plotly.subplots import make_subplots



# predefine color pallette alias

cnf = 'grey' # confirmed

dth = 'red' # death

rec = 'lightgreen' # recovered

act = 'orange' # active

!ls ../input/covid-forecasting-datasets/
train_df = pd.read_csv('../input/covid19-global-forecasting-week-3/train.csv')

test_df = pd.read_csv('../input/covid19-global-forecasting-week-3/test.csv')
lockdown = pd.read_csv("../input/covid-forecasting-datasets/countryLockdowndates.csv")

lockdown.columns = ['Country_Region', 'Province_State', 'Date_lockdown', 'Type', 'Reference']

lowckdown_df = train_df.merge(lockdown, on=['Country_Region', 'Province_State'], how='left')
measures = pd.read_csv("../input/covid-forecasting-datasets/acaps-covid-19-government-measures-dataset.csv")

measures.columns = ['id', 'Country_Region', 'iso', 'admin_level_name', 'pcode', 'region',

       'category', 'measure', 'targeted_pop_group', 'comments',

       'measures_date_implemented', 'source', 'source_type', 'link', 'entry_date',

       'alternative_source']



measures = measures[[

    'id', 'Country_Region', 'category', 'measure', 'targeted_pop_group', 'comments', 'measures_date_implemented'

    ]]



measures["Country_Region"].replace({

    "United States of America": "US",

    "Russia": "Russian Federation",

    "Viet Nam": "Vietnam",

    "Korea Republic of": "Korea, South",

    "Czech Republic":"Korea, South"

    }, inplace=True)
measures['category'] = measures.category.str.lower()

measures['measure'] =  measures.measure.str.lower()
measures['measures_date_implemented'] = pd.to_datetime(measures["measures_date_implemented"], errors='coerce')



measures_country = measures.groupby(['Country_Region', 'measure']).agg({'measures_date_implemented': 'min'}).reset_index()





pivot_measures = pd.pivot_table(measures_country, values='measures_date_implemented', index=['Country_Region'],

                    columns=['measure'], aggfunc='min')

pivot_measures = pivot_measures.reset_index()

pivot_measures



lowckdown_df.Date = pd.to_datetime(lowckdown_df.Date)

lockdown_measure_df = lowckdown_df.merge(

    pivot_measures, left_on=['Country_Region'], right_on=['Country_Region'], how='left')

lockdown_measure_df





for column in pivot_measures.columns.tolist():

    if column in ['Country_Region', 'measures_date_implemented']:

        continue

    lockdown_measure_df.loc[lockdown_measure_df['Date'] >= pd.to_datetime(lockdown_measure_df[column]), column+'_flag'] = 1

    lockdown_measure_df.drop(columns=[column], inplace=True)



lockdown_measure_df.head()



lockdown_measure_df.fillna(0.0, inplace=True)
location_global = pd.read_csv("../input/covid-forecasting-datasets/time_series_covid19_confirmed_global.csv")[['Province/State', 'Country/Region', 'Lat', 'Long']

                                                                  ].rename(columns={

                    'Province/State': 'Province_State',

                    'Country/Region': 'Country_Region'

})



location_us = pd.read_csv("../input/covid-forecasting-datasets/time_series_covid19_confirmed_US.csv")[['Province_State', 'Country_Region', 'Lat', 'Long_']

                                                                  ].rename(columns={

                    'Long_': 'Long'

})



location = location_global.append(location_us)





location = location[(location.Lat != 0) & (location.Long != 0)].drop_duplicates(

    ['Province_State', 'Country_Region'])



location





lockdown_geo = lockdown_measure_df.merge(

    location.fillna(0.0), on=['Country_Region', 'Province_State'], how='left')
lockdown_geo['Location'] = lockdown_geo['Province_State'].astype(str) + '_' + lockdown_geo['Country_Region'].astype(str)



from sklearn import preprocessing



types = lockdown_geo.dtypes

cat_columns = [t[0] for t in types.iteritems() if ((t[1] not in ['int64', 'float64']))]



print('Label encoding categorical columns:', cat_columns)

encoders = {}

Locations = []

for col in cat_columns:

    lbl = preprocessing.LabelEncoder()

    if col == 'Location':

        Locations += lockdown_geo[col].unique().tolist()

    if col == 'Date':

        continue

    lockdown_geo[col] = lbl.fit_transform(lockdown_geo[col].astype(str))

    encoders[col] = lbl

    

all_dates = list(set(train_df.Date.unique().tolist() + test_df.Date.unique().tolist()))

lbl = preprocessing.LabelEncoder()

lbl.fit(all_dates)

lockdown_geo['Date'] = lbl.transform(lockdown_geo[['Date']].astype(str))
train_window_size = len(lockdown_geo.Date.unique())

lookback_window_size = 14

forecast_window_size = 33
features_set = []

labels_cases = []

labels_fatalities = []



from tqdm import *



for i in tqdm(range(lookback_window_size, train_window_size - forecast_window_size)):



    for location in lockdown_geo.Location.unique().tolist():

        df = lockdown_geo[lockdown_geo.Location == location].reset_index()

        features_set.append(df.iloc[i-lookback_window_size:i, 3:].values)

        labels_cases.append(df.iloc[i:i+forecast_window_size, :]['ConfirmedCases'])

        labels_fatalities.append(df.iloc[i:i+forecast_window_size, :]['Fatalities'])

        

# test_features_set = []

# test_labels_cases = []

# test_labels_fatalities = []

 

# for i in tqdm(range(66, 67)):

#     for location in lockdown_geo.Location.unique().tolist():#encoders['Location'].transform(['New York_US']).tolist():#l

# #         print(location)

#         df = lockdown_geo[lockdown_geo.Location == location].reset_index()

#         test_features_set.append(df.iloc[i-7:i, 3:].values)

#         test_labels_cases.append(df.iloc[i:i+7, :]['ConfirmedCases'])

#         test_labels_fatalities(df.iloc[i:i+7, :]['Fatalities'])

        

# future_features_set = []

# i=73

# for location in lockdown_geo.Location.unique().tolist():#encoders['Location'].transform(['New York_US']).tolist():#l

#     df = lockdown_geo[lockdown_geo.Location == location].reset_index()

#     future_features_set.append(df.iloc[i-7:i, 3:].values)



# future_features_set = pd.np.array(future_features_set)
future_features_set_1 = []

i = train_window_size

for location in lockdown_geo.Location.unique().tolist():

    df = lockdown_geo[lockdown_geo.Location == location].reset_index()

    future_features_set_1.append(df.iloc[i-lookback_window_size:i, 3:].values)

#         labels_cases.append(df.iloc[i:i+forecast_window_size, :]['ConfirmedCases'])

#         labels_fatalities.append(df.iloc[i:i+forecast_window_size, :]['Fatalities'])



future_features_set_1 = pd.np.array(future_features_set_1)
future_features_set_2 = []

i = lbl.transform([test_df.Date.min()])[0]

for location in lockdown_geo.Location.unique().tolist():

    df = lockdown_geo[lockdown_geo.Location == location].reset_index()

    future_features_set_2.append(df.iloc[i-lookback_window_size:i, 3:].values)

#         labels_cases.append(df.iloc[i:i+forecast_window_size, :]['ConfirmedCases'])

#         labels_fatalities.append(df.iloc[i:i+forecast_window_size, :]['Fatalities'])



future_features_set_2 = pd.np.array(future_features_set_2)
test_df.fillna(0.0, inplace=True)

test_df['Location'] = test_df['Province_State'].astype(str) + '_' + test_df['Country_Region'].astype(str)

test_df['Location'] = encoders['Location'].transform(test_df['Location'])

test_df.head()
import gc

gc.collect()
features_set, labels_cases, labels_fatalities = pd.np.array(features_set), pd.np.array(labels_cases), pd.np.array(labels_fatalities)

# test_features_set, test_labels = pd.np.array(test_features_set), pd.np.array(test_labels)

size = features_set.shape[0]

gc.collect()



split = int(size*(9/10))

values = pd.np.nan_to_num(features_set)

n_train_time = 365*24



train = values[:split, :, :]

test = values[split:, :, :]

train_y_cases = labels_cases[:split]

test_y_cases = labels_cases[split:]



train_y_fatalities = labels_fatalities[:split]

test_y_fatalities = labels_fatalities[split:]
from keras.models import Sequential

from keras.layers import Dense

from keras.layers import LSTM

from keras.layers import Dropout



model_cases = Sequential()

model_cases.add(LSTM(units=100,input_shape=(features_set.shape[1], features_set.shape[2])))

model_cases.add(Dropout(0.2))

model_cases.add(Dense(units = 64, activation='relu'))

model_cases.add(Dropout(0.2))

model_cases.add(Dense(units = 32, activation='relu'))

model_cases.add(Dropout(0.2))

model_cases.add(Dense(units = forecast_window_size))

model_cases.compile(optimizer = 'adam', loss = 'mean_squared_error')

history_cases = model_cases.fit(train, train_y_cases, epochs = 10, batch_size = 32, validation_data=(test, test_y_cases))
model_fatalities = Sequential()

model_fatalities.add(LSTM(units=100,input_shape=(features_set.shape[1], features_set.shape[2])))

model_fatalities.add(Dropout(0.2))

model_fatalities.add(Dense(units = 64, activation='relu'))

model_fatalities.add(Dropout(0.2))

model_fatalities.add(Dense(units = 32, activation='relu'))

model_fatalities.add(Dropout(0.2))

model_fatalities.add(Dense(units = forecast_window_size))

model_fatalities.compile(optimizer = 'adam', loss = 'mean_squared_error')

history_fatalities = model_fatalities.fit(train, train_y_fatalities, epochs = 10, batch_size = 32, validation_data=(test, test_y_fatalities))
forecast_cases_future_1 = model_cases.predict(future_features_set_1)

forecast_fatalities_future_1 = model_fatalities.predict(future_features_set_1)



forecast_cases_future_2 = model_cases.predict(future_features_set_2)

forecast_fatalities_future_2 = model_fatalities.predict(future_features_set_2)
date_1 = future_features_set_1[:,13,1]

date_2 = future_features_set_2[:,13,1]

locations_1 =  future_features_set_1[:,13,-1]

locations_2 =  future_features_set_2[:,13,-1]
location_col = []

Date_col = []

predictions_col = []

fatalities_col = []



for i,pred in enumerate(forecast_cases_future_1.tolist()):

    location_col += [locations_1[i]]*33

    Date_col += [date_1[i]+x+1 for x in range(33)]

    predictions_col += pred

    

for i,pred in enumerate(forecast_fatalities_future_1.tolist()):

    fatalities_col += pred

  

    

df_1 = pd.DataFrame({'Location': location_col, 'Date': Date_col, 'predictions_1': predictions_col, 'fatalities_1': fatalities_col})



location_col = []

Date_col = []

predictions_col = []

fatalities_col = []



for i,pred in enumerate(forecast_cases_future_2.tolist()):

    location_col += [locations_2[i]]*33

    Date_col += [date_2[i]+x+1 for x in range(33)]

    predictions_col += pred

    

for i,pred in enumerate(forecast_fatalities_future_2.tolist()):

#     location_col += [locations_1[i]]*33

#     Date_col += [date_1[i]]*33

    fatalities_col += pred

      

df_2 = pd.DataFrame({'Location': location_col, 'Date': Date_col, 'predictions_2': predictions_col, 'fatalities_2': fatalities_col})
df_1['Date'] = df_1['Date']
encoders
test_df['Date'] =lbl.transform(test_df['Date'])
test_df
df_2.Location = df_2.Location.astype('int')

df_2.Date = df_2.Date.astype('int')



df_1.Location = df_1.Location.astype('int')

df_1.Date = df_1.Date.astype('int')
df_1.loc[df_1.predictions_1 < 0, 'predictions_1'] = 0

df_1.loc[df_1.fatalities_1 < 0, 'fatalities_1'] = 0

df_2.loc[df_2.predictions_2 < 0, 'predictions_2'] = 0

df_2.loc[df_2.fatalities_2 < 0, 'fatalities_2'] = 0
df_2
t = test_df.merge(

    df_2, on=['Location', 'Date'], how='left'

    ).merge(df_1, on=['Location', 'Date'], how='left')
t['ConfirmedCases'] = t.predictions_1

t['Fatalities'] = t.fatalities_1
t.loc[t.ConfirmedCases.isnull(), 'ConfirmedCases'] = t[t.ConfirmedCases.isnull()].predictions_2

t.loc[t.Fatalities.isnull(), 'Fatalities'] = t[t.Fatalities.isnull()].fatalities_2
4038 - 3126
t.fillna(0, inplace=True)
t['Date'] = lbl.inverse_transform(t['Date'])

t.head()
t['ConfirmedCases'] = t.ConfirmedCases.astype(int)

t['Fatalities'] = t.Fatalities.astype(int)

t
t[['ForecastId', 'ConfirmedCases', 'Fatalities']].to_csv("submission.csv", index=False)
# dates1 = [i+1+date_1 for i in range(33)]

# dates2 = [i+1+date_2 for i in range(33)]
# dates = future_features_set_2[:,13,1]

# locations =  future_features_set_1[:,13,-1]
# forecast_cases_future_1.shape
forecast_cases_future

forecast_cases_future[forecast_cases_future < 0] = 0

forecast_fatalities_future[forecast_fatalities_future < 0] = 0
# df.iloc[:,3:]
# test_dates = []

# for item in (test_features_set[:,:,1] + 7 ).tolist():

#     test_dates+=item

    

# test_locations = []

# for item in (test_features_set[:,:,-1]).tolist():

#     test_locations+=item



# len(test_locations)



# test_forecast = []

# for item in out.tolist():

#     test_forecast+=item

# len(test_forecast)
# future_out = model.predict(future_features_set)
# for item in (future_features_set[:,:,1] + 7 ).tolist():

#     test_dates+=item

    

# for item in (future_features_set[:,:,-1]).tolist():

#     test_locations+=item



# for item in future_out.tolist():

#     test_forecast+=item

# len(test_forecast)
# test_labels_list = []

# for item in test_labels.tolist():

#     test_labels_list+=item

# len(test_labels_list)



# test_labels_list += [pd.np.nan]*2142

# len(test_labels_list)



# # test_locations = test_features_set[:,6,-1].tolist()

# # test_dates = test_features_set[:,6,1].tolist()

# locs = encoders['Location'].inverse_transform([int(i) for i in test_locations]).tolist()

# # [loc for loc in locs if '_US' in loc]



# dates = test_dates#encoders['Date'].inverse_transform([int(i) for i in test_dates]).tolist()

# # [loc for loc in locs if '_US' in loc]



# Forecast = pd.DataFrame({'Forecasted_cases': test_forecast, 'location': locs, 'Date': dates, 'True_cases': test_labels_list})



# US_Forecast = Forecast[Forecast.location.astype(str).str.contains('_US')]







# fig = go.Figure()

# for location in ['0.0_Spain', 'Washington_US', 'Michigan_US', 'New York_US']:#US_Forecast.location.unique().tolist():

#     location_df = Forecast[Forecast.location == location]

# #     country_df = lockdown_geo[lockdown_geo.Country_Region == country].groupby(['Date']).agg({'Fatalities': 'sum'}).reset_index()

#     fig.add_trace(go.Scatter(x=location_df.Date, y=location_df.True_cases,

#                         mode='lines+markers',

#                         name=location+' - True Cases'))



#     fig.add_trace(go.Scatter(x=location_df.Date, y=location_df.Forecasted_cases,

#                         mode='lines+markers',

#                         name=location+' - Forecasted Cases'))





#     # for business_id in ['e0CTLPxTnFEQSqQ1FJUqog', 'dWFUKB_HPBIE87AFBHEb_w', 'CMN3KmB5SEfONN00s2nEeQ', '7MNBIoGznDHhC1AfxGWOFw']:

#     #     plot_df = bad_business_df[bad_business_df.business_id == business_id]

#     #     plot_df = plot_df[['stars']].resample("M").mean().reset_index()

#     #     plot_df = plot_df[plot_df.date_time > "2014-01-01"]

#     #     fig.add_trace(go.Scatter(x=plot_df.date_time, y=plot_df.stars,

#     #                         mode='lines+markers',

#     #                         name=bad_names_dict[business_id]))



#     fig.update_layout(

#         title={

#             'text': "Confirmed cases over time"})



# fig.show()