import pandas as pd

import numpy as np

np.random.seed(1337) # for reproducibility



import seaborn as sns

from datetime import datetime

import matplotlib.pyplot as plt

from os import listdir, remove

from os.path import isfile, join



base_folder = '/kaggle/input/'

plt.rcParams['figure.figsize'] = [15, 7]



# let's define different sets of features

ext_cols = ['LaborForceTotal', 'LaborForcePerCapita', 'DeathRate', 'AirTrafficPassengersTotal',

            'AirTrafficPassengersPerCapita', 'HospitalBedDensity', 'Obesity', 'OldPeople',

            'PhysiciansDensity', 'AlcoholConsumptionPerCapita', 'CigaretteConsumptionPerCapita']

# ext_cols = ['LaborForcePerCapita', 'DeathRate', 'AirTrafficPassengersPerCapita', 'HospitalBedDensity',

#             'Obesity', 'OldPeople', 'PhysiciansDensity']

pop_cols = ['Yearly change', 'Density', 'Land Area', 'Med. Age', 'Urban Pop', 'Population']

# pop_cols = ['Yearly change', 'Med. Age', 'Urban Pop', 'Density']

add_cols = ['DayNum', 'PreviousDay-0ExposedDensity']



# loading day-by-day data (based on hopkins datasets) prepared by Kaggle

# (it contains 'ConfirmedCases' and 'Fatalities')

data_base = base_folder + 'covid19-global-forecasting-week-4/'

df = pd.read_csv(data_base + 'train.csv').drop(columns=['Id'])

df.rename(columns={'Province_State': 'Province/State', 'Country_Region': 'Country/Region'}, inplace=True)



# fill empty Province

df['Province/State'].fillna('entire country', inplace=True)



# add Delta features

df['ConfirmedCasesDelta'] = df.groupby(['Country/Region', 'Province/State'])['ConfirmedCases'].diff().fillna(0)

df['FatalitiesDelta'] = df.groupby(['Country/Region', 'Province/State'])['Fatalities'].diff().fillna(0)





# set the proper type for Date column and calculate DayNum

df['Date'] = pd.to_datetime(df['Date']).dt.date

day_zero = min(df['Date'])

df['DayNum'] = (df['Date'] - day_zero).apply(lambda x: int(x.days))

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

TAIL = 50

previous_days_cols = add_extra_features_from_previous_days(df, TAIL)

df
df['PreviousDay-0ConfirmedCases'].max()
def special_day_CC(org_df, number_of_cc):

    # calculating for every country days from the day a patient # was confirmed

    print('calculating for every Country & Province days passed from the first day when ConfirmedCases >= {}'.format(number_of_cc))

    col_final = 'Day_CC{}'.format(number_of_cc)

    col = 'Day_CC{}_zero'.format(number_of_cc)

    org_df[col] = org_df.where(

        (org_df['ConfirmedCases'] >= number_of_cc) &

        ((org_df['PreviousDay-0ConfirmedCases'] < number_of_cc)|(org_df['Date'] == day_zero))

    )['DayNum']

    org_df[col] = org_df.groupby(['Country/Region', 'Province/State'])[col].ffill()

    org_df[col] = org_df.groupby(['Country/Region', 'Province/State'])[col].bfill()

    # calculating real DayNum counted from the day "zero"

    day_num = org_df['DayNum'] - org_df[col] + 1

    org_df[col_final] = (day_num - day_num.where(day_num<0).fillna(0)).fillna(0)

    return col, col_final



def special_day_F(org_df, number_of_f):

    # calculating for every country days from the day a patient # was confirmed

    print('calculating for every Country & Province days passed from the first day when Fatalities >= {}'.format(number_of_f))

    col_final = 'Day_F{}'.format(number_of_f)

    col = 'Day_F{}_zero'.format(number_of_f)

    org_df[col] = org_df.where(

        (org_df['Fatalities'] >= number_of_f) & 

        ((org_df['PreviousDay-0Fatalities'] < number_of_f)|(org_df['Date'] == day_zero))

    )['DayNum']

    org_df[col] = org_df.groupby(['Country/Region', 'Province/State'])[col].ffill()

    org_df[col] = org_df.groupby(['Country/Region', 'Province/State'])[col].bfill()

    # calculating real DayNum counted from the day "zero"

    day_num = org_df['DayNum'] - org_df[col] + 1

    org_df[col_final] = (day_num - day_num.where(day_num<0).fillna(0)).fillna(0)

    return col, col_final

special_cols1 = []

special_cols2 = []



c1, c2 = special_day_CC(df, 1)

special_cols1.append(c1)

special_cols2.append(c2)

c1, c2 = special_day_CC(df, 50)

special_cols1.append(c1)

special_cols2.append(c2)

c1, c2 = special_day_CC(df, 200)

special_cols1.append(c1)

special_cols2.append(c2)

c1, c2 = special_day_CC(df, 500)

special_cols1.append(c1)

special_cols2.append(c2)

c1, c2 = special_day_CC(df, 1000)

special_cols1.append(c1)

special_cols2.append(c2)

c1, c2 = special_day_CC(df, 5000)

special_cols1.append(c1)

special_cols2.append(c2)

c1, c2 = special_day_CC(df, 20000)

special_cols1.append(c1)

special_cols2.append(c2)

c1, c2 = special_day_CC(df, 50000)

special_cols1.append(c1)

special_cols2.append(c2)

c1, c2 = special_day_CC(df, 100000)

special_cols1.append(c1)

special_cols2.append(c2)



c1, c2 = special_day_F(df, 1)

special_cols1.append(c1)

special_cols2.append(c2)

c1, c2 = special_day_F(df, 25)

special_cols1.append(c1)

special_cols2.append(c2)

c1, c2 = special_day_F(df, 50)

special_cols1.append(c1)

special_cols2.append(c2)

c1, c2 = special_day_F(df, 200)

special_cols1.append(c1)

special_cols2.append(c2)

c1, c2 = special_day_F(df, 500)

special_cols1.append(c1)

special_cols2.append(c2)

c1, c2 = special_day_F(df, 1000)

special_cols1.append(c1)

special_cols2.append(c2)

c1, c2 = special_day_F(df, 2000)

special_cols1.append(c1)

special_cols2.append(c2)

c1, c2 = special_day_F(df, 5000)

special_cols1.append(c1)

special_cols2.append(c2)

c1, c2 = special_day_F(df, 10000)

special_cols1.append(c1)

special_cols2.append(c2)



special_cols2
# loading population data 

df_population = pd.read_csv(base_folder + 'worldpopulaton-ver2/all_population.csv', delimiter=';', decimal=',', na_values='N.A.')

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

    ('Myanmar (Burma)', 'Burma'),

    ('East Timor', 'Timor-Leste'),

    ('Macedonia', 'North Macedonia'),

    ('Cape Verde', 'Cabo Verde'),

    ('Congo (Republic)', 'Congo (Brazzaville)'),

    ('Congo (Democratic Republic)', 'Congo (Kinshasa)'),

    ('Palestinian Territories', 'West Bank and Gaza'),

    ('United Kingdom of Great Britain and Northern Ireland', 'United Kingdom'),

    ('Vatican City', 'Holy See'),

    ('Sao Tome & Principe', 'Sao Tome and Principe')

]



country_state_pairs_to_replace = [

    (('Greenland', 'entire country'), ('Denmark', 'Greenland')),

    (('Anguilla', 'entire country'), ('United Kingdom', 'Anguilla')),

    (('Bermuda', 'entire country'), ('United Kingdom', 'Bermuda')),

    (('British Virgin Islands', 'entire country'), ('United Kingdom', 'British Virgin Islands')),

    (('Isle of Man', 'entire country'), ('United Kingdom', 'Isle of Man')),

    (('Turks and Caicos Islands', 'entire country'), ('United Kingdom', 'Turks and Caicos Islands')),

    (('Sint Maarten', 'entire country'), ('Netherlands', 'Sint Maarten')),

    (('Saint Pierre & Miquelon', 'entire country'), ('France', 'Saint Pierre and Miquelon')),

    (('Falkland Islands', 'entire country'), ('United Kingdom', 'Falkland Islands (Malvinas)'))

    

]

# loading different datasets from World By Map

csv_dir = base_folder + 'worldbymap-ver2/'

files = [

    'labor_force',

    'death_rate',

    'air_traffic_passengers',

    'hospital_bed_density',

    'obesity',

    'old_people',

    'physicians_density',

    'cigarettes',

    'alcohol'

]

wbm = {}

for f in files:

    wbm[f] = pd.read_csv(csv_dir + f + '.csv', delimiter=';', decimal=',', na_values='N.A.')

    for ctr in countries_to_replace:

        wbm[f]['Country'] = wbm[f]['Country'].replace(ctr[0], ctr[1])

    for pair in country_state_pairs_to_replace:

        ind = (wbm[f]['Country'] == pair[0][0]) & (wbm[f]['State'] == pair[0][1])

        wbm[f].loc[ind, 'Country'] = pair[1][0]

        wbm[f].loc[ind, 'State'] = pair[1][1]

wbm[files[0]]
df_add = pd.DataFrame()

for dataset in wbm.keys():

    if df_add.shape == (0, 0):

        df_add = wbm[dataset].copy()

    else:

        df_add = pd.merge(df_add, wbm[dataset], on=['Country', 'State'], how='left')

df_add.rename(columns={"Country": "Country/Region", "State": "Province/State"}, inplace=True)

df_add
# merging available external data

df_external = pd.merge(df_population, df_add, on=['Country/Region', 'Province/State'], how='left')



def fill_missing_percapita_values(dfr, feature_total, feature_percapita):

    cond = (dfr[feature_percapita].isna()) & (df_external[feature_total].notna()) & (df_external['Population'].notna())

    ind = df_external[cond].index

    df_external.loc[ind, feature_percapita] = df_external.loc[ind, feature_total] / df_external.loc[ind, 'Population'] * 100.0



# now we have to fill LaborForcePerCapita and AirTrafficPassengersPerCapita for some regions where total values have been given only  

fill_missing_percapita_values(df_external, 'LaborForceTotal', 'LaborForcePerCapita')

fill_missing_percapita_values(df_external, 'AirTrafficPassengersTotal', 'AirTrafficPassengersPerCapita')



# filling NaNs in external data with column means

df_external[pop_cols+ext_cols] = df_external[pop_cols+ext_cols].apply(lambda x: x.fillna(x.mean()),axis=0)



# changing names of some countries to



df_external
# merging covid dataset with additional external data

df_pop = pd.merge(df, df_external, on=['Country/Region', 'Province/State'], how='left')

df_pop
cond_ctry = [

#     ((df['Country/Region']=='Poland') & (df['Province/State']=='entire country'), 'red'),

    ((df['Country/Region']=='Germany') & (df['Province/State']=='entire country'), 'blue'),

    ((df['Country/Region']=='China') & (df['Province/State']=='Hubei'), 'green'),

    ((df['Country/Region']=='Italy') & (df['Province/State']=='entire country'), 'cyan'),

    ((df['Country/Region']=='Spain') & (df['Province/State']=='entire country'), 'magenta'),

    ((df['Country/Region']=='Korea, South') & (df['Province/State']=='entire country'), 'gray'),

]

start_day = 'Day_F25'

cond_day = df[start_day]>0

feature = 'Fatalities'



for i, cnd in enumerate(cond_ctry):

    my_df = df_pop[(cnd[0]) & (cond_day)] 

    my_x = my_df[start_day]

    my_y = my_df[feature]/my_df['PreviousDay-11ConfirmedCases']



    chart_data = pd.DataFrame({

        'x': my_x,

        'y': my_y})



    sns.lineplot(x='x', y='y', data=chart_data, color=cnd[1])
# With new datasets coming from Hopkins, some country names may change.

# Uncomment 2 lines below and check if there are countries with missing population data.

ccc = pop_cols + ext_cols

df_pop[df_pop[ccc].isnull().any(axis=1)][['Country/Region', 'Province/State'] + ccc].drop_duplicates(subset=['Country/Region', 'Province/State'])

# adding column "ExposedDensity" - population still exposed to covid per km2

# 1.43 is my own factor for the exponential function indicating hidden spread of COVID (people having COVID but never diagnosed)

df_pop['ExposedDensity'] = (df_pop['Population'] - np.power(df_pop['ConfirmedCases'], 1.43))/df_pop['Land Area']

density = df_pop.groupby(['Country/Region', 'Province/State'])['Density']

df_pop['PreviousDay-0ExposedDensity'] = df_pop.groupby(['Country/Region', 'Province/State'])['ExposedDensity'].shift(periods=1, fill_value=np.nan)

df_pop['PreviousDay-0ExposedDensity'] = df_pop.apply(

    lambda row: row['Density'] if np.isnan(row['PreviousDay-0ExposedDensity']) else row['PreviousDay-0ExposedDensity'],

    axis=1

)

df_pop
# from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_absolute_error, mean_squared_log_error

from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, PowerTransformer

from scipy import stats



model_x_columns_without_dummies = add_cols + pop_cols + ext_cols + previous_days_cols + special_cols2

model_x_columns = model_x_columns_without_dummies + continent_columns



def train_test_split(X, y, test_size=0.3, random_state=0):

    day_first = min(X['DayNum'])

    day_last = max(X['DayNum'])

    number_of_days_for_train = int(round((day_last-day_first+1)*(1-test_size),0))

    last_day_for_training = number_of_days_for_train + day_first - 1

    X_tr = X[X['DayNum']<=last_day_for_training].copy()

    y_tr = y[X['DayNum']<=last_day_for_training].copy()

    X_te = X[X['DayNum']>last_day_for_training].copy()

    y_te = y[X['DayNum']>last_day_for_training].copy()

    return X_tr, X_te, y_tr, y_te



# let's define an evaluation metric

def rmsle(ytrue, ypred):

    return np.sqrt(mean_squared_log_error(ytrue, ypred))



def mae(ytrue, ypred):

    return mean_absolute_error(ytrue, ypred)



def mse(ytrue, ypred):

    return mean_squarred_error(ytrue, ypred)



def analyse3(tr_true, tr_pred, val_true, val_pred, test_true, test_pred):

    chart_data0 = pd.DataFrame({

        'x0': tr_true.flatten(),

        'x1': tr_pred.flatten(),

        'y': tr_true.flatten()-tr_pred.flatten()})



    chart_data1 = pd.DataFrame({

        'x0': val_true.flatten(),

        'x1': val_pred.flatten(),

        'y': val_true.flatten()-val_pred.flatten()})

    

    chart_data2 = pd.DataFrame({

        'x0': test_true.flatten(),

        'x1': test_pred.flatten(),

        'y': test_true.flatten()-test_pred.flatten()})

    

    fig, ax =plt.subplots(1,3)

    sns.scatterplot(x='x0', y='y', data=chart_data0, color='black', ax=ax[0])

    sns.scatterplot(x='x1', y='y', data=chart_data0, color='red', ax=ax[0])

    sns.scatterplot(x='x0', y='y', data=chart_data1, color='black', ax=ax[1])

    sns.scatterplot(x='x1', y='y', data=chart_data1, color='red', ax=ax[1])

    sns.scatterplot(x='x0', y='y', data=chart_data2, color='black', ax=ax[2])

    sns.scatterplot(x='x1', y='y', data=chart_data2, color='red', ax=ax[2])

    

    print('MAE train: {}'.format(round(rmsle(tr_true, tr_pred), 6)))

    print('MAE val:  {}'.format(round(rmsle(val_true, val_pred), 6)))

    print('MAE test:  {}'.format(round(rmsle(test_true, test_pred), 6)))



def prepare_data(df, what_to_predict, test_size=0.3, dropna=False):

    df_tmp = df.copy()

    

    if dropna:

        df_tmp.dropna(inplace=True)

    

    df_tmp.loc[df_tmp[what_to_predict]<0, what_to_predict] = 0

    # preparing X and y datasets for output model training

    data_X = df_tmp[model_x_columns+['Country/Region']]

    data_y = df_tmp[what_to_predict].values.flatten()

    # splitting data to train and test

    return train_test_split(data_X, data_y, test_size=test_size, random_state=42)

    

def predict_output(input_data, model):

    y_pred = np.abs(model.predict(input_data))

    return y_pred



def expm1_relu(y):

    tmp = np.expm1(y)

    tmp[tmp<0]=0    

    return np.around(tmp)
scaler0 = None

scaler1 = None

scaler2 = None



def scale_data(data):

    global scaler0, scaler1, scaler2

    data_bis = data.copy()

    daynum = data_bis['DayNum'].copy()

    memory = dict()

    for c in special_cols2:

        memory[c] = data_bis[c].copy()

        

    if scaler1:

        data_bis[model_x_columns_without_dummies] = scaler0.transform(data[model_x_columns_without_dummies])

        data_bis[model_x_columns_without_dummies] = scaler1.transform(data[model_x_columns_without_dummies])

        data_bis[model_x_columns_without_dummies] = scaler2.transform(data_bis[model_x_columns_without_dummies])



    else:

        scaler0 = PowerTransformer()

        scaler1 = MinMaxScaler()

        scaler2 = StandardScaler()

        data_bis[model_x_columns_without_dummies] = scaler0.fit_transform(data[model_x_columns_without_dummies])

        data_bis[model_x_columns_without_dummies] = scaler1.fit_transform(data[model_x_columns_without_dummies])

        data_bis[model_x_columns_without_dummies] = scaler2.fit_transform(data_bis[model_x_columns_without_dummies])



    for c in memory.keys():

        data_bis[c] = memory[c]

    data_bis['DayNum'] = daynum

    

    return data_bis



df_pop_bis = scale_data(df_pop)

df_pop_bis



# ccc = pop_cols + ext_cols + special_cols2

# df_pop[df_pop[ccc].isnull().any(axis=1)][['Country/Region', 'Province/State'] + pop_cols + ext_cols]
# neural network

from keras.models import Sequential, load_model

from keras.layers import Dense, Dropout, Activation, ELU

from keras.metrics import mean_squared_error, mean_absolute_error, accuracy

from keras.optimizers import Adam, SGD, RMSprop

from keras.callbacks import ModelCheckpoint

# from keras import regularizers

from keras.regularizers import l1, l2, l1_l2



model_path = join('.')

model_file_f = join(model_path, 'nn_model_f.h5')

model_file_cc = join(model_path, 'nn_model_cc.h5')
# swish activation function

from keras import backend as K

from keras.utils.generic_utils import get_custom_objects

from keras.activations import sigmoid



class Swish(Activation):

    

    def __init__(self, activation, **kwargs):

        super(Swish, self).__init__(activation, **kwargs)

        self.__name__ = 'swish'



def swish(x, beta = 0.6):

    return (x * sigmoid(beta * x))



get_custom_objects().update({'swish': Swish(swish)})
model_f = Sequential()

model_f.add(Dense(50, input_dim=len(model_x_columns)-1, activation='swish'))

model_f.add(Dropout(0.2))

model_f.add(Dense(15, activation='elu'))

model_f.add(Dropout(0.2))

model_f.add(Dense(1, activation='elu'))

opt_f = Adam(learning_rate=0.001, beta_1=0.94, beta_2=0.99, amsgrad=False)

model_f.compile(loss='mean_absolute_error', optimizer=opt_f, metrics=['mae'])



save_best_only_callback_f = ModelCheckpoint(

    filepath=model_file_f,

    monitor='val_loss',

    verbose=1,

    save_best_only=True,

    save_weights_only=False,

    mode='min',

    period=1

)



data_X_tr, data_X_rest, data_y_tr, data_y_rest = prepare_data(df_pop_bis, 'FatalitiesDelta', test_size=0.3, dropna=False)

data_X_val, data_X_test, data_y_val, data_y_test = train_test_split(data_X_rest, data_y_rest, test_size=0.5, random_state=111)



history = model_f.fit(x=data_X_tr[model_x_columns].drop(columns=['DayNum']), y=data_y_tr,

                      validation_data=(data_X_val[model_x_columns].drop(columns=['DayNum']), data_y_val),

                      epochs=170, batch_size=128, verbose=1, callbacks=[save_best_only_callback_f])



# summarize history for loss

plt.plot(history.history['loss'][0:])

plt.plot(history.history['val_loss'][0:])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'val'], loc='upper left')

plt.show()

model_f = load_model(model_file_f)



tr_pred = predict_output(data_X_tr[model_x_columns].drop(columns=['DayNum']), model_f)

val_pred = predict_output(data_X_val[model_x_columns].drop(columns=['DayNum']), model_f)

test_pred = predict_output(data_X_test[model_x_columns].drop(columns=['DayNum']), model_f)

# print(data_y_test)

analyse3(data_y_tr, tr_pred,

         data_y_val, val_pred,

         data_y_test, test_pred)
model_cc = Sequential()

model_cc.add(Dense(28, input_dim=len(model_x_columns)-1, activation='swish'))

model_cc.add(Dropout(0.0))

model_cc.add(Dense(15, activation='elu'))

model_cc.add(Dropout(0.0))

model_cc.add(Dense(1, activation='swish'))



opt_cc = Adam(learning_rate=0.0001, beta_1=0.988, beta_2=0.99, amsgrad=False)



model_cc.compile(loss='mean_absolute_error', optimizer=opt_cc, metrics=['mae'])



save_best_only_callback_cc = ModelCheckpoint(

    filepath=model_file_cc,

    monitor='val_loss',

    verbose=1,

    save_best_only=True,

    save_weights_only=False,

    mode='min',

    period=1

)



data_X_tr, data_X_rest, data_y_tr, data_y_rest = prepare_data(df_pop_bis, 'ConfirmedCasesDelta', test_size=0.3, dropna=False)

data_X_val, data_X_test, data_y_val, data_y_test = train_test_split(data_X_rest, data_y_rest, test_size=0.5, random_state=111)



history = model_cc.fit(x=data_X_tr[model_x_columns].drop(columns=['DayNum']), y=data_y_tr,

                       validation_data=(data_X_val[model_x_columns].drop(columns=['DayNum']), data_y_val),

                       epochs=770, batch_size=128, verbose=1, callbacks=[save_best_only_callback_cc])



# summarize history for loss

plt.plot(history.history['loss'][0:])

plt.plot(history.history['val_loss'][0:])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'val'], loc='upper left')

plt.show()
model_cc = load_model(model_file_cc)



tr_pred = predict_output(data_X_tr[model_x_columns].drop(columns=['DayNum']), model_cc)

val_pred = predict_output(data_X_val[model_x_columns].drop(columns=['DayNum']), model_cc)

test_pred = predict_output(data_X_test[model_x_columns].drop(columns=['DayNum']), model_cc)

# print(data_y_test)

analyse3((data_y_tr), (tr_pred),

         (data_y_val), (val_pred),

         (data_y_test), (test_pred))
aaa = dict()

chart_data = dict()

countries = ['US', 'France', 'Korea, South', 'Spain', 'Poland', 'Italy', 'Japan', 'Iran']

colors = ['red', 'blue', 'orange', 'black', 'cyan', 'brown', 'grey', 'purple']



for ind, (c, color) in enumerate(zip(countries, colors)):

    tmp = df_pop_bis.where(df_pop_bis['Country/Region']==c).dropna(subset=['DayNum'])

    aaa[c] = tmp[tmp['Day_CC1']>0]

    chart_data[c] = pd.DataFrame({

        'x': aaa[c]['Day_CC1'],

        'y': np.log1p(aaa[c]['Fatalities'])})



    sns.lineplot(x='x', y='y', data=chart_data[c], color=color)
# prepare test data

df_test = pd.read_csv(data_base + 'test.csv')

df_test.rename(columns={'Province_State': 'Province/State', 'Country_Region': 'Country/Region'}, inplace=True)



# replace empty province

df_test['Province/State'].fillna('entire country', inplace=True)



# set proper type for the Date column and calculate DayNum

df_test['Date'] = pd.to_datetime(df_test['Date']).dt.date

df_test['DayNum'] = (df_test['Date'] - day_zero).apply(lambda x: int(x.days))



# get countries' special days from df train dataset, join them with the test dataset and set the counter for each such special day

for c1, c2 in zip(special_cols1, special_cols2):

    zero_days = pd.DataFrame(df.groupby(['Country/Region', 'Province/State', c1]).size().reset_index()[['Country/Region', 'Province/State', c1]])

    zero_days.drop_duplicates(subset=['Country/Region', 'Province/State'], keep='last', inplace=True)

    df_test = df_test.merge(zero_days, on=['Country/Region', 'Province/State'], how='left')

    real_day_num = df_test['DayNum'] - df_test[c1] + 1

    df_test[c2] = (real_day_num - real_day_num.where(real_day_num<0).fillna(0)).fillna(0)

df_test
# merging df_test with population data

df_test_pop = pd.merge(df_test, df_external, on=['Country/Region', 'Province/State'], how='left')

df_test_pop
# let's take available data from training dataset (overlap with test dataset)

output_columns = ['ConfirmedCases', 'Fatalities']

tmp_output_columns = ['ConfirmedCases_y', 'Fatalities_y']



last_training_day = df['DayNum'].max()

first_test_day = df_test['DayNum'].min()

train_test_keys = ['Country/Region', 'Province/State', 'DayNum']

df_test_pop_train = pd.merge(df_test_pop, df_pop[df_pop['DayNum']>=first_test_day][train_test_keys + ['PreviousDay-0ExposedDensity', 'ExposedDensity'] + previous_days_cols + output_columns],

                             on=train_test_keys, how='left')

df_test_pop_train



# ccc = pop_cols + ext_cols + special_cols2

# let's check if there are some missing data now

# df_test_pop[df_test_pop[ccc].isnull().any(axis=1)][train_test_keys + pop_cols + ext_cols]
# let's stick training and test datasets (we need it to have previous days info)

df_test_final = pd.concat([df_pop[df_pop['DayNum']<first_test_day], df_test_pop_train]).reset_index(drop=True)

df_test_final[(df_test_final['Country/Region']=='Poland')&(df_test_final['DayNum']<=last_training_day+1)&(df_test_final['DayNum']>last_training_day-10)]
# we need do keep some feature without scaling to calculation ExposedDensity

df_test_final['PopulationOrg'] = df_test_final['Population'].copy()

df_test_final['Land Area Org'] = df_test_final['Land Area'].copy()
# final loop to predict every day separately and to make feature engineering on-the-fly

model_cc = load_model(model_file_cc)

model_f = load_model(model_file_f)



last_test_day = df_test['DayNum'].max()

for day in range(last_training_day+1, last_test_day+1):

    print('predicting day {} ({} to go)'.format(day, last_test_day-day))

    # calculate columns for previous days

    add_extra_features_from_previous_days(df_test_final, TAIL)

    # keep unscaled previous day output data

    df_test_final['PreviousDay-0ConfirmedCases_NotScaled'] = df_test_final['PreviousDay-0ConfirmedCases'].copy()

    df_test_final['PreviousDay-0Fatalities_NotScaled'] = df_test_final['PreviousDay-0Fatalities'].copy()

    # add ExposedDensity

    df_test_final['PreviousDay-0ExposedDensity'] = df_test_final.groupby(['Country/Region', 'Province/State'])['ExposedDensity'].shift(periods=1, fill_value=np.nan)

    df_test_final['PreviousDay-0ExposedDensity'] = df_test_final.apply(

        lambda row: row['Density'] if np.isnan(row['PreviousDay-0ExposedDensity']) else row['PreviousDay-0ExposedDensity'],

        axis=1

    ) 

    # get current day only

    current_day = df_test_final[df_test_final['DayNum']==day].copy()

    # scale data

    current_day_scaled = scale_data(current_day)

    # predict output for the current day

    current_day_scaled['ConfirmedCasesDelta'] = predict_output(current_day_scaled[model_x_columns].drop(columns=['DayNum']), model_cc)

    current_day_scaled['FatalitiesDelta'] = predict_output(current_day_scaled[model_x_columns].drop(columns=['DayNum']), model_f)

    current_day_scaled['ConfirmedCases'] = current_day_scaled['PreviousDay-0ConfirmedCases_NotScaled'] + current_day_scaled['ConfirmedCasesDelta']

    current_day_scaled['Fatalities'] = current_day_scaled['PreviousDay-0Fatalities_NotScaled'] + current_day_scaled['FatalitiesDelta']

    # fill ExposedDensity

    current_day_scaled['ExposedDensity'] = (current_day_scaled['PopulationOrg'] - np.power(current_day_scaled['ConfirmedCases'], 1.43))/current_day_scaled['Land Area Org']



    # fill df_test with current day predictions

    cond = df_test_final['DayNum']==day

    df_test_final.loc[cond, output_columns+['ExposedDensity']] = current_day_scaled[output_columns+['ExposedDensity']].copy()

    df_test_final.loc[(cond)&(df_test_final[cond]['ConfirmedCases']<current_day_scaled['ConfirmedCases']), 'ConfirmedCases'] = df_test_final['PreviousDay-0ConfirmedCases']

    df_test_final.loc[(cond)&(df_test_final[cond]['Fatalities']<current_day_scaled['Fatalities']), 'Fatalities'] = df_test_final['PreviousDay-0Fatalities']



df_test_final.columns

submission_columns = ['ForecastId', 'ConfirmedCases', 'Fatalities']

# convert to int

df_test_final.loc[df_test_final['ForecastId'].isna(), 'ForecastId'] = 0

df_test_final[submission_columns] = df_test_final[submission_columns].astype(int)

# save submission

df_test_final[df_test_final['DayNum']>=first_test_day][submission_columns].to_csv('submission.csv', index=False)

# submission view

df_test_final[df_test_final['DayNum']>=last_training_day][submission_columns]
df_test_final['ConfirmedCases'].max()
df_test_final['Fatalities'].max()