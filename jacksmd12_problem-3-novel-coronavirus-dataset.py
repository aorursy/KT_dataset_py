import pandas as pd

import numpy as np

import plotly.express as px

import os

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MinMaxScaler

from sklearn.neighbors import KNeighborsRegressor

from sklearn.metrics import mean_squared_log_error



from plotly.subplots import make_subplots

import plotly.graph_objects as go
# readh in datasets:

data_dir = '/kaggle/input/novel-corona-virus-2019-dataset'

confirmed = pd.read_csv(os.path.join(data_dir, 'time_series_covid_19_confirmed.csv'))

deaths = pd.read_csv(os.path.join(data_dir, 'time_series_covid_19_deaths.csv'))

recovered = pd.read_csv(os.path.join(data_dir, 'time_series_covid_19_recovered.csv'))
print('confirmed data shape: ' + str(confirmed.shape) )

print('deaths data shape: ' + str(deaths.shape))

print('recovered data shape: ' + str(recovered.shape))
# print(confirmed.columns == deaths.columns)

# print(confirmed.columns == recovered.columns)
confirmed_melt = pd.melt(confirmed, id_vars=['Province/State', 'Country/Region', 'Lat', 'Long'],

                   var_name='Date', value_name='Confirmed')

deaths_melt = pd.melt(deaths, id_vars=['Province/State', 'Country/Region', 'Lat', 'Long'],

                     var_name='Date', value_name='Deaths')

recovered_melt = pd.melt(recovered, id_vars=['Province/State', 'Country/Region', 'Lat', 'Long'],

                        var_name='Date', value_name='Recovered')
print(confirmed_melt.shape)

print(deaths_melt.shape)

print(recovered_melt.shape)
df =confirmed_melt.merge(deaths_melt, how='outer', on=['Province/State', 'Country/Region', 'Lat', 'Long', 'Date'])

df = df.merge(recovered_melt, how='left', on=['Province/State', 'Country/Region', 'Lat', 'Long', 'Date'])
print('the following country does not have recovery info: ' + np.unique(df[df['Recovered'].isna()]['Country/Region']))
# since the number of country without recovery is relatively small, just drop them

df = df[df['Recovered'].isna() != True]
df_melt = pd.melt(df, id_vars = ['Province/State', 'Country/Region', 'Lat', 'Long', 'Date'])

fig = px.scatter_geo(df_melt, lat='Lat', lon='Long', size='value', color='variable',

                    animation_frame='Date', projection='natural earth')

fig.show()
# convert to weekly average 

df_weekly = df

df_weekly.index = pd.to_datetime(df_weekly['Date'])

df_weekly = df_weekly.drop(['Province/State', 'Date', 'Lat', 'Long'], axis=1)

df_weekly = df_weekly.groupby(['Country/Region']).resample('W').mean()

df_weekly = df_weekly.reset_index()



# adding 1 to the confirmed cases to get around divide by 0 error before the spread

df_weekly['Death_Rate'] = df_weekly['Deaths'] / (df_weekly['Confirmed'] + 1)



df_weekly['Date'] = df_weekly['Date'].dt.strftime('%Y-%m-%d')
random_sample_country = pd.Series(np.unique(df_weekly['Country/Region'])).sample(10, random_state=5)



# always have US

random_sample_country = random_sample_country.append(pd.Series(['US']))



df_weekly = df_weekly[df_weekly['Country/Region'].isin(random_sample_country)]

fig = px.scatter(df_weekly, x='Confirmed', y='Death_Rate', 

                 color='Country/Region',

                 range_x=[min(df_weekly['Confirmed']), max(df_weekly['Confirmed'])],

                 range_y=[min(df_weekly['Death_Rate']), max(df_weekly['Death_Rate'])],

                 animation_group='Country/Region', animation_frame='Date')



fig.update_traces(marker_size=20, opacity = 0.7)

fig.show()
df_weekly = df

df_weekly.index = pd.to_datetime(df_weekly['Date'])

df_weekly = df_weekly.groupby(['Country/Region']).resample('W').mean()

df_weekly = df_weekly.reset_index()
df_knn = df_weekly



# create new feature for 'days'

df_knn['days'] = (df_knn.loc[:, 'Date'].apply(pd.Timestamp) - pd.Timestamp(df_knn.loc[:,'Date'].min())).dt.days



fig = px.scatter(df_knn, x = 'days', y = 'Country/Region', size='Confirmed', color='Country/Region')

fig.show()



df_knn = df_knn.reset_index(drop=True)
# rescaling using MinMax method

X = df_knn.loc[:, ['Lat', 'Long', 'days']]

cols = X.columns

scaler = MinMaxScaler()

scaler.fit(X)

X = pd.DataFrame(scaler.transform(X), columns = cols)



# # rescaling using normalize

# X = df_svr.loc[:, ['Lat', 'Long', 'days']]

# cols = X.columns

# from sklearn.preprocessing import normalize

# X = normalize(X)



y = df_knn.loc[:, 'Confirmed']



# break into training and test data set

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=15)



# KNN model

neigh = KNeighborsRegressor(n_neighbors=2, weights='distance', algorithm='brute', p=2)

neigh.fit(X_train, y_train)

pred_train = neigh.predict(X_train)

pred_test = neigh.predict(X_test)



RMSLE_train = np.sqrt(mean_squared_log_error(y_train, pred_train))

RMSLE_test = np.sqrt(mean_squared_log_error(y_test, pred_test))



print('Train data RMSLE: ' + str(RMSLE_train))

print('Test data RMSLE: ' + str(RMSLE_test))

print('')
# method for finding the growth rate for n-days. Defaults to start at day 0, with US coordinates



def confirmed_growth_rate(day_n, start_day=0, Lat = 37.0902, Long = -95.7129):

    day_n_X = scaler.transform(pd.DataFrame([[Lat, Long, day_n]]))

    day_n_X = pd.DataFrame(day_n_X, columns=cols)

    pred_n_cases = neigh.predict(day_n_X)

    

    start_X = scaler.transform(pd.DataFrame([[Lat, Long, start_day]]))

    start_X = pd.DataFrame(start_X, columns=cols)

    pred_start_cases = neigh.predict(start_X)

    

    # check for divide by 0 problem

    if pred_start_cases == 0:

        pred_start_cases = 1

    

    growth_rate = (pred_n_cases - pred_start_cases) / pred_start_cases

    return growth_rate



day_range = np.arange(0, 30)

growth_rate = []

interval = 15 # days





for day in day_range:

    start_day = day

    day_n = day + interval

    growth_rate.append(confirmed_growth_rate(day_n, start_day)[0])



# confirmed_growth_rate(day_n=15, start_day=14)
fig = px.scatter(x=day_range, y=growth_rate)

fig.show()
# explore relationship between confirmed, deaths and recovered

cases = df_knn[['Confirmed', 'Deaths', 'Recovered']]

cases_melt = pd.melt(cases, id_vars='Confirmed')



fig = px.scatter(cases_melt, x='Confirmed', y='value', facet_col='variable')

fig.show()
# subset data to predict recovery rate

X = df_knn[['Lat', 'Long', 'days']]

cols = X.columns

scaler = MinMaxScaler()

scaler.fit(X)

X = pd.DataFrame(scaler.transform(X), columns = cols)



y = df_knn[['Recovered']]



# break into training and test data set

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=15)

# KNN model

neigh = KNeighborsRegressor(n_neighbors=2, weights='distance', algorithm='brute', p=2)

neigh.fit(X_train, y_train)

pred_train = neigh.predict(X_train)

pred_test = neigh.predict(X_test)



RMSLE_train = np.sqrt(mean_squared_log_error(y_train, pred_train))

RMSLE_test = np.sqrt(mean_squared_log_error(y_test, pred_test))



print('Train data RMSLE: ' + str(RMSLE_train))

print('Test data RMSLE: ' + str(RMSLE_test))

print('')
# method for finding the recovery rate for n-days. Defaults to start at day 0, with US coordinates



def recovery_rate(day_n, start_day=0, Lat = 37.0902, Long = -95.7129):

    day_n_X = scaler.transform(pd.DataFrame([[Lat, Long, day_n]]))

    day_n_X = pd.DataFrame(day_n_X, columns=cols)

    pred_n_cases = neigh.predict(day_n_X)

    

    start_X = scaler.transform(pd.DataFrame([[Lat, Long, start_day]]))

    start_X = pd.DataFrame(start_X, columns=cols)

    pred_start_cases = neigh.predict(start_X)

    

    # check for divide by 0 problem

    if pred_start_cases == 0:

        pred_start_cases = 1

    

    recovery_rate = (pred_n_cases - pred_start_cases) / pred_start_cases

    return recovery_rate
day_range = np.arange(0, 30)

recover_rate = []

interval = 15 # days





for day in day_range:

    start_day = day

    day_n = day + interval

    recover_rate.append(recovery_rate(day_n, start_day)[0])





# recovery_rate(day_n=15, start_day=14)
fig = px.scatter(x = day_range, y= recover_rate)

fig.show()