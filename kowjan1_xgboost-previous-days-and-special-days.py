import pandas as pd

import numpy as np



import seaborn as sns

from datetime import datetime

import matplotlib.pyplot as plt

from os import listdir

from os.path import isfile, join



base_folder = '/kaggle/input/'



# loading day-by-day data (based on hopkins datasets) prepared by Kaggle

# (it contains 'ConfirmedCases' and 'Fatalities', but doesn't contain 'Recovered')

data_base = base_folder + 'covid19-global-forecasting-week-2/'

df = pd.read_csv(data_base + 'train.csv')

df.rename(columns={'Province_State': 'Province/State', 'Country_Region': 'Country/Region'}, inplace=True)

df['Province/State'].fillna('entire country', inplace=True)



df
def add_extra_features_from_previous_days(data_fr, tail_size=5):

    cols_tmp = []

    col_prefix = 'PreviousDay'

    for i in range (0, tail_size):

        col_cc = '{}-{}ConfirmedCases'.format(col_prefix, i)

        col_f  = '{}-{}Fatalities'.format(col_prefix, i)



        data_fr[col_cc] = data_fr.groupby(['Country/Region', 'Province/State'])['ConfirmedCases'].shift(periods=i+1, fill_value=0)

        data_fr[col_f] = data_fr.groupby(['Country/Region', 'Province/State'])['Fatalities'].shift(periods=i+1, fill_value=0)

        data_fr[col_cc + 'Delta'] = data_fr.groupby(['Country/Region', 'Province/State'])[col_cc].diff().fillna(0)

        data_fr[col_f + 'Delta'] = data_fr.groupby(['Country/Region', 'Province/State'])[col_f].diff().fillna(0)

        cols_tmp += [col_cc, col_f, col_cc + 'Delta', col_f + 'Delta']

    # df['PreviousDay-0ConfirmedCases'] = df.groupby(['Country/Region', 'Province/State'])['ConfirmedCases'].shift(periods=1, fill_value=0)

    return  cols_tmp



# creating extra features from the history: previous day, previous day -1, previous day -2 ...

TAIL = 28

previous_days_cols = add_extra_features_from_previous_days(df, TAIL)

df
df['PreviousDay-0ConfirmedCases'].max()
# calculating day "zero" for every country

day_zero = datetime.strptime(min(df['Date']), '%Y-%m-%d')

df['DayNum'] = (df['Date'].astype('datetime64[ns]') - day_zero).apply(lambda x: int(x.days))

df['DayZero'] = df.where((df['ConfirmedCases'] > 0) & ((df['PreviousDay-0ConfirmedCases'] == 0)|(df['Date'] == day_zero)))['DayNum']

df['DayZero'] = df.groupby(['Country/Region', 'Province/State'])['DayZero'].ffill()

df['DayZero'] = df.groupby(['Country/Region', 'Province/State'])['DayZero'].bfill()

# calculating real DayNum counted from the day "zero"

real_day_num = df['DayNum'] - df['DayZero'] + 1

df['RealDayNum'] = real_day_num - real_day_num.where(real_day_num<0).fillna(0)

df
# loading population data 

df_population = pd.read_csv(base_folder + 'world-populaton/all_population.csv', delimiter=';', decimal=',', na_values='N.A.')

# urban population: NaNs with 100% (it's a good estimation!)

df_population['Urban Pop'] = df_population['Urban Pop'].fillna(100.0)

# OHE for a continent

df_population = pd.get_dummies(df_population, columns=['Continent'])

# let's remember new columns for continents

continent_columns = []

for c in df_population.columns:

    if 'Continent_' in c:

        continent_columns.append(c)

df_population
# Countries names map between World By Map and Hopking datasets

countries_to_replace = [

    ('Czech Republic', 'Czechia'),

    ('United States of America', 'US'),

    ('Côte d\'Ivoire (Ivory Coast)', 'Côte d\'Ivoire'),

    ('Korea (South)', 'Korea, South'),

    ('Swaziland', 'Eswatini'),

    ('Gambia', 'The Gambia'),

    ('Myanmar (Burma)', 'Myanmar'),

    ('East Timor', 'Timor-Leste'),

    ('Macedonia', 'North Macedonia'),

    ('Cape Verde', 'Cabo Verde'),

    ('Congo (Republic)', 'Congo (Brazzaville)'),

    ('Congo (Democratic Republic)', 'Congo (Kinshasa)'),

    ('Palestinian Territories', 'State of Palestine'),

    ('Bahamas', 'The Bahamas'),

    ('United Kingdom of Great Britain and Northern Ireland', 'United Kingdom'),

    ('Vatican City', 'Holy See')

]

# loading different datasets from World By Map

csv_dir = base_folder + 'worldbymap/'

files = [

    'labor_force',

    'death_rate',

    'air_traffic_passengers',

    'hospital_bed_density',

    'obesity',

    'old_people',

    'physicians_density'

]

wbm = {}

for f in files:

    wbm[f] = pd.read_csv(csv_dir + f + '.csv', delimiter=';', decimal=',', na_values='N.A.')

    for ctr in countries_to_replace:

        wbm[f] = wbm[f].replace(ctr[0], ctr[1])

wbm[files[0]]
df_add = pd.DataFrame()

for dataset in wbm.keys():

    if df_add.shape == (0, 0):

        df_add = wbm[dataset].copy()

    else:

        df_add = df_add.merge(wbm[dataset], on='Country', how='left')

df_add.rename(columns={"Country": "Country/Region"}, inplace=True)

df_add
df_external = pd.merge(df_population, df_add, on='Country/Region', how='left')

# df_external = df_population.copy()
# merging covid dataset with additional external data

df_pop = pd.merge(df, df_external, on=['Country/Region', 'Province/State'], how='left')

df_pop
from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_absolute_error, mean_squared_log_error

from sklearn.preprocessing import MinMaxScaler

from scipy import stats



ext_cols = ['LaborForceTotal', 'LaborForcePerCapita ', 'DeathRate', 'AirTrafficPassengersTotal',

            'AirTrafficPassengersPerCapita', 'HospitalBedDensity', 'Obesity', 'OldPeople',

            'PhysiciansDensity']

# ext_cols = []

pop_cols = ['DayNum', 'RealDayNum', 'Yearly change', 'Density', 'Land Area', 'Med. Age', 'Urban Pop', 'Population']

pop_cols = ['DayNum', 'RealDayNum', 'Med. Age', 'Urban Pop']



model_x_columns_without_dummies = pop_cols + ext_cols + previous_days_cols

model_x_columns = model_x_columns_without_dummies + continent_columns



# let's define an evaluation metric

def rmsle(ytrue, ypred):

    return np.sqrt(mean_squared_log_error(ytrue, ypred))



def mae(ytrue, ypred):

    return mean_absolute_error(ytrue, ypred)



# checking how the model predicts test data

def analyse(data_y_test, data_y_pred):

    chart_data = pd.DataFrame({'x1': data_y_test.flatten(),

                               'x2': data_y_pred.flatten(),

                               'y': np.abs(data_y_test.flatten()-data_y_pred.flatten()).flatten()})

    sns.scatterplot(x='x1', y='y', data=chart_data, color='black')

    sns.scatterplot(x='x2', y='y', data=chart_data, color='red')

    print('RMSLE: {}'.format(round(rmsle(data_y_test, data_y_pred), 6)))



def analyse2(tr_y, tr_pred, data_y_test, data_y_pred):

    chart_data0 = pd.DataFrame({

        'x00': tr_y.flatten(),

        'x01': tr_pred.flatten(),

        'y0': np.abs(tr_y.flatten()-tr_pred.flatten()).flatten()})



    chart_data1 = pd.DataFrame({

        'x10': data_y_test.flatten(),

        'x11': data_y_pred.flatten(),

        'y1': np.abs(data_y_test.flatten()-data_y_pred.flatten()).flatten()})

    

    fig, ax =plt.subplots(1,2)

    sns.scatterplot(x='x00', y='y0', data=chart_data0, color='blue', ax=ax[0])

    sns.scatterplot(x='x01', y='y0', data=chart_data0, color='yellow', ax=ax[0])

    sns.scatterplot(x='x10', y='y1', data=chart_data1, color='black', ax=ax[1])

    sns.scatterplot(x='x11', y='y1', data=chart_data1, color='red', ax=ax[1])

    

    print('RMSLE train: {}'.format(round(rmsle(tr_y, tr_pred), 6)))

    print('RMSLE test:  {}'.format(round(rmsle(data_y_test, data_y_pred), 6)))





def prepare_data(df, what_to_predict, test_size=0.3, dropna=False):

    df_tmp = df.copy()

    

    if dropna:

        df_tmp.dropna(inplace=True)

        

    # preparing X and y datasets for output model training

    data_X = df_tmp[model_x_columns]

    data_y = np.log1p(df_tmp[[what_to_predict]].values.flatten())

    # splitting data to train and test

    return train_test_split(data_X, data_y, test_size=test_size, random_state=42)

    

def predict_output(input_data, model):

    df_final = input_data[model_x_columns].copy()

    y_pred = model.predict(df_final)

    return y_pred



def expm1_relu(y):

    tmp = np.expm1(y)

    tmp[tmp<0]=0    

    return np.around(tmp)
# xgboost

# n_estimators: 336

# max_depth: 5

# min_child_weight: 1

# gamma: 0.16

# subsample: 0.98

# colsample_bytree: 0.86



from xgboost.sklearn import XGBRegressor



data_X_tr, data_X_test, data_y_tr, data_y_test = prepare_data(df_pop, 'Fatalities', test_size=0.7, dropna=True)

# data_X_val, data_X_test, data_y_val, data_y_test = train_test_split(data_X_rest, data_y_rest, test_size=0.5, random_state=111)



model_f = XGBRegressor(learning_rate=0.01, n_estimators=880, max_depth=3, min_child_weight=0.0, gamma=0.0,

                       subsample=0.8, colsample_bytree=0.7, reg_alpha=0.0, reg_lambda=0.0,

                       objective='reg:squarederror', scale_pos_weight=1, seed=37)



model_f.fit(data_X_tr, data_y_tr)



tr_pred = predict_output(data_X_tr, model_f)



data_y_pred = predict_output(data_X_test, model_f)

analyse2(expm1_relu(data_y_tr), expm1_relu(tr_pred), expm1_relu(data_y_test), expm1_relu(data_y_pred))
model_cc = XGBRegressor(learning_rate=0.1, n_estimators=100, max_depth=5, min_child_weight=0, gamma=0,

                        subsample=0.7, colsample_bytree=0.8, reg_alpha=0, reg_lambda=0,

                        objective='reg:squaredlogerror', scale_pos_weight=1, seed=37, colsample_bynode=0.5)



data_X_tr, data_X_test, data_y_tr, data_y_test = prepare_data(df_pop, 'ConfirmedCases', test_size=0.7, dropna=True)

# data_X_val, data_X_test, data_y_val, data_y_test = train_test_split(data_X_rest, data_y_rest, test_size=0.5, random_state=111)



hist = model_cc.fit(data_X_tr, data_y_tr)
tr_pred = predict_output(data_X_tr, model_cc)



data_y_pred = predict_output(data_X_test, model_cc)

analyse2(expm1_relu(data_y_tr), expm1_relu(tr_pred), expm1_relu(data_y_test), expm1_relu(data_y_pred))
# prepare test data

df_test = pd.read_csv(data_base + 'test.csv')

df_test.rename(columns={'Province_State': 'Province/State', 'Country_Region': 'Country/Region'}, inplace=True)



# replace empty province

df_test['Province/State'].fillna('entire country', inplace=True)

# calculate daynum based on the date of report

df_test['DayNum'] = (df_test['Date'].astype('datetime64[ns]') - day_zero).apply(lambda x: int(x.days))

# get countries' zero days from df train dataset and join them with the test dataset

zero_days = pd.DataFrame(df.groupby(['Country/Region', 'Province/State', 'DayZero']).size().reset_index()[['Country/Region', 'Province/State', 'DayZero']])

zero_days.drop_duplicates(subset=['Country/Region', 'Province/State'], keep='last', inplace=True)

df_test = df_test.merge(zero_days, on=['Country/Region', 'Province/State'], how='left')



df_test
# calculating RealDayNum based on DayZero

real_day_num = df_test['DayNum'] - df_test['DayZero'] + 1

df_test['RealDayNum'] = real_day_num - real_day_num.where(real_day_num<0).fillna(0)



# merging df_test with population data

df_test_pop = pd.merge(df_test, df_external, on=['Country/Region', 'Province/State'], how='left')

df_test_pop.fillna(df_test_pop.mean(), inplace=True)

df_test_pop
output_columns = ['ConfirmedCases', 'Fatalities']

tmp_output_columns = ['ConfirmedCases_y', 'Fatalities_y']



# let's take available data a from training dataset (overlap with test dataset)

last_training_day = df['DayNum'].max()

first_test_day = df_test['DayNum'].min()

train_test_keys = ['Country/Region', 'Province/State', 'DayNum']

df_test_pop = pd.merge(df_test_pop, df[df['DayNum']>=first_test_day][train_test_keys+output_columns], on=train_test_keys, how='left')

df_test_pop[output_columns] = df_test_pop[output_columns].fillna(0).copy()

df_test_pop
# And now comes a very importsnt part of the prediction.

# As we need to use previous prediction in the next prediction we have to predict day by day

last_test_day = df_test['DayNum'].max()

for day in range(last_training_day+1, last_test_day+1):

    print('predicting day {} ({} to go)'.format(day, last_test_day-day))

    up_to_current_day = df_test_pop.where(df_test['DayNum']<=day).dropna(subset=['Country/Region'])

    # calculate columns for previous days

    previous_days_columns = add_extra_features_from_previous_days(up_to_current_day, TAIL)

    # predict output for current day

    up_to_current_day['ConfirmedCases'] = expm1_relu(predict_output(up_to_current_day[model_x_columns], model_cc))

    up_to_current_day['Fatalities'] = expm1_relu(predict_output(up_to_current_day[model_x_columns], model_f))

    # fill df_test with current day predictions

    tmp_dataset = up_to_current_day[up_to_current_day['DayNum']==day][train_test_keys+output_columns]

    df_test_pop = pd.merge(df_test_pop, tmp_dataset, on=train_test_keys, how='left', suffixes=('', '_y'))

    df_test_pop[tmp_output_columns] = df_test_pop[tmp_output_columns].fillna(0).copy()

    df_test_pop['ConfirmedCases'] += df_test_pop['ConfirmedCases_y']

    df_test_pop['Fatalities'] += df_test_pop['Fatalities_y']

    df_test_pop.drop(columns=tmp_output_columns, inplace=True)

# up_to_current_day_scaled[model_x_columns]
up_to_current_day[up_to_current_day.isnull().any(axis=1)]
submission_columns = ['ForecastId', 'ConfirmedCases', 'Fatalities']

df_test_pop[submission_columns].to_csv('submission.csv', index=False)

df_test_pop
df_test_pop['ConfirmedCases'].max()
df_test_pop['Fatalities'].max()