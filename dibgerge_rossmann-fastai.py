# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session



import numpy as np

import pandas as pd

from os.path import splitext, join

from IPython.display import HTML, display

import os

import re

import time

from isoweek import Week

from functools import partial



from sklearn.preprocessing import StandardScaler

import tensorflow as tf

from tensorflow import keras

from tensorflow.keras.layers import Dense, Embedding, concatenate, Flatten, Dropout



pd.set_option('display.max_columns', 1000)
data = {}

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        full_name = os.path.join(dirname, filename)

        key = splitext(filename)[0]        

        data[key] = pd.read_csv(full_name, low_memory=False)

    

display(f'Loaded file keys: {list(data.keys())}', )
def add_date_parts(df, date_column, parts=None, prefix=None):

    """

    Add date information to the dataframe inplace.

    """

    prefix = prefix or ''

    if parts is None:

        parts = [

            'Year',

            'Month',

            'Week',

            'Day',

            'Dayofweek',

            'Dayofyear',

            'Is_month_end',

            'Is_month_start',

            'Is_quarter_end',

            'Is_quarter_start',

            'Is_year_end',

            'Is_year_start',

            'Elapsed'

        ]

    if not np.issubdtype(df[date_column].dtype, np.datetime64):

        df[date_column] = pd.to_datetime(df[date_column], infer_datetime_format=True)

    

    s = df[date_column]

    for part in parts:

        if part == 'Week':

            df[prefix + part] = s.dt.isocalendar()['week']

        elif part == 'Elapsed':

            df[prefix + part] = s.astype(np.int64) // 10 ** 9

        else:

            df[prefix + part] = getattr(s.dt, part.lower())
googletrend = data['googletrend']

googletrend['Date'] = pd.to_datetime(googletrend.week.str.split(' - ', expand=True)[0])

googletrend['State'] = googletrend.file.str.split('_', expand=True)[2]

googletrend.loc[googletrend.State=='NI', 'State'] = 'HB,NI'

add_date_parts(googletrend, 'Date', parts=['Year', 'Week'])

googletrend.head()
trend_de = googletrend[googletrend.file == 'Rossmann_DE'][['trend', 'Year', 'Week']]

trend_de.head()
weather = data['weather'].merge(data['state_names'], how='left', left_on='file', right_on='StateName').drop('file', axis=1)

add_date_parts(weather, 'Date', parts=['Year', 'Week'])

weather.head()
store = data['store'].merge(data['store_states'], how='left', on='Store')

store.head()
add_date_parts(data['train'], 'Date')

add_date_parts(data['test'], 'Date')
def merge_all(df):

    out = (df

        .merge(store, how='left', on='Store')

        .merge(googletrend, how='left', on=['State', 'Year', 'Week'], suffixes=('', '_y'))

        .merge(trend_de, how='left', on=['Year', 'Week'], suffixes=('', '_DE'))

        .merge(weather, how='left', on=['State', 'Date'], suffixes=('', '_y'))

    )

    # Drop replicated columns for right merged tables and a couple unwanted ones

    drop_cols = list(out.columns[out.columns.str.endswith('_y')]) + ['week', 'file']

    out.drop(drop_cols, inplace=True, axis=1)

    

    # Check if the merge resulted in any new nulls

    print('Merge has nulls:', any([

        any(out.StoreType.isnull()),

        any(out.trend.isnull()),

        any(out.trend_DE.isnull()),

        any(out.Mean_TemperatureC.isnull()),

    ]))

    return out



train = merge_all(data['train'])

test = merge_all(data['test'])
%%time



def add_features(df):

    df = df.copy()

    # Convert StateHoliday to a flag (0/1) value instead of a categorical one

    df.StateHoliday = (df.StateHoliday != '0').astype(int)

    

    # Add the number of months a competition has been open

    df['CompetitionOpenSinceYear'] = df.CompetitionOpenSinceYear.fillna(1900).astype(np.int32)

    df['CompetitionOpenSinceMonth'] = df.CompetitionOpenSinceMonth.fillna(1).astype(np.int32)

    df['CompetitionOpenSince'] = pd.to_datetime(dict(year=df.CompetitionOpenSinceYear, month=df.CompetitionOpenSinceMonth, day=15))

    df['CompetitionDaysOpen'] = df.Date.subtract(df.CompetitionOpenSince).dt.days

    df.loc[df.CompetitionDaysOpen < 0, "CompetitionDaysOpen"] = 0

    df.loc[df.CompetitionOpenSinceYear < 1990, "CompetitionDaysOpen"] = 0

    df['CompetitionMonthsOpen'] = (df.CompetitionDaysOpen//30).clip(0, 24)

    

    # Add the number of weeks since promotion 

    df['Promo2SinceYear'] = df.Promo2SinceYear.fillna(1900).astype(np.int32)

    df['Promo2SinceWeek'] = df.Promo2SinceWeek.fillna(1).astype(np.int32)

    df['Promo2Since'] = pd.to_datetime([

        Week(y[0], int(y[1])).monday() for y in df[['Promo2SinceYear', 'Promo2SinceWeek']].values

    ])

    df['Promo2Days'] = df.Date.subtract(df.Promo2Since).dt.days

    df.loc[df.Promo2Days < 0, 'Promo2Days'] = 0

    df['Promo2Weeks'] = (df['Promo2Days']//7).clip(0, 25)

    

    # Add elapsed time since and to next of the following flags

    columns = ['SchoolHoliday', 'StateHoliday', 'Promo']



    df.sort_values(['Store', 'Date'], inplace=True)

    for column in columns:

        mask = df[column] == 1    

        for name, method in zip([f'After{column}', f'Before{column}'], ['ffill', 'bfill']):

            df.loc[mask, name] = df.loc[mask, 'Date']

            df[name] = df.groupby('Store')[name].fillna(method=method)

            df[name] = (df.Date - df[name]).dt.days.fillna(0).astype(int)

        

    # Set the active index to Date, so we can do rolling sums

    df.set_index('Date', inplace=True)

    

    # We will sum total number of the following in last/next week

    bw = df.sort_index().groupby('Store')[columns].rolling(7, min_periods=1).sum()    

    fw = df.sort_index(ascending=False).groupby('Store')[columns].rolling(7, min_periods=1).sum()

    df = (

        df

        .merge(bw, how='left', on=['Store', 'Date'], suffixes=['', '_bw'])

        .merge(fw, how='left', on=['Store', 'Date'], suffixes=['', '_fw'])

    )

    return df.reset_index()



train = add_features(train)

test = add_features(test)
train.reset_index().to_feather('/kaggle/working/train')

test.to_feather('/kaggle/working/test')
train = pd.read_feather('/kaggle/working/train')

test = pd.read_feather('/kaggle/working/test')
categorical_cols = [

    'Store','DayOfWeek', 'Year', 'Month', 

    'Day', 'StateHoliday', 'CompetitionMonthsOpen','Promo2Weeks',

    'StoreType', 'Assortment', 'PromoInterval', 'CompetitionOpenSinceYear',

    'Promo2SinceYear', 'State', 'Week', 'Events',

    'Promo_fw', 'Promo_bw', 'StateHoliday_fw', 'StateHoliday_bw',

    'SchoolHoliday_fw', 'SchoolHoliday_bw'

]

continuous_cols = [

    'CompetitionDistance', 'Max_TemperatureC', 'Mean_TemperatureC', 'Min_TemperatureC',

    'Max_Humidity', 'Mean_Humidity', 'Min_Humidity', 'Max_Wind_SpeedKm_h', 

    'Mean_Wind_SpeedKm_h', 'CloudCover', 'trend', 'trend_DE',

    'AfterStateHoliday', 'BeforeStateHoliday', 'Promo', 'SchoolHoliday'

]

target = 'Sales'



all_cols = categorical_cols + continuous_cols



train = train[train.Sales > 0]



# Select those specified columns

train = train[all_cols + [target, 'Date']]

test = test[all_cols + ['Date', 'Id']]
# Convert categorical columns to numbers

for c in categorical_cols:    

    train[c] = train[c].astype('category').cat.as_ordered()

    test[c] = test[c].astype('category').cat.as_ordered()

    test[c].cat.set_categories(train[c].cat.categories, ordered=True, inplace=True)

    

    train[c] = train[c].cat.codes + 1

    test[c] = test[c].cat.codes + 1

    

for c in continuous_cols:

    train[c] = train[c].fillna(0).astype('float32')

    test[c] = test[c].fillna(0).astype('float32')



scaler = StandardScaler()

train[continuous_cols] = scaler.fit_transform(train[continuous_cols])

test[continuous_cols] = scaler.transform(test[continuous_cols])
def get_data(has_validation=True):

    # train here access a global value, copies it and stores it locally (to the function)

    data = train.copy().set_index('Date')

    X = data[all_cols]

    y = np.log(data[target])

    y_max = y.max()

    y_min = y.min()

    

    if has_validation:        

        split_date = '2015-06-15'

        val_split_date = '2015-06-16'



        X_train, X_val = X.loc[:split_date], X.loc[val_split_date:]



        # Now, convert the training data into a list 

        X_train = [X_train[continuous_cols].values] + [X_train[c].values[..., None] for c in categorical_cols]

        X_val = [X_val[continuous_cols].values] + [X_val[c].values[..., None] for c in categorical_cols]



        # Get the labels

        y_train, y_val = y.loc[:split_date].values, y.loc[val_split_date:].values

        

        y_train = (y_train - y_min)/(y_max - y_min)

        y_val = (y_val - y_min)/(y_max - y_min)

        return y_max, y_min, X_train, y_train, X_val, y_val

    else:

        X_train = [X[continuous_cols].values] + [X[c].values[..., None] for c in categorical_cols]

        y_train = y.values

        y_train = (y_train - y_min)/(y_max - y_min)

        return y_max, y_min, X_train, y_train





def get_rmspe(y_min, y_max):

    def rmspe(y_true, y_pred):

        y_true = tf.math.exp(y_true * (y_max - y_min) + y_min)

        y_pred = tf.math.exp(y_pred * (y_max - y_min) + y_min)

        return tf.math.sqrt(tf.reduce_mean(tf.square((y_true - y_pred)/y_true)))

    return rmspe





def get_model(y_min, y_max, learning_rate=1e-3, dropout=0.1):

    # get the embedding sizes

    embedding_map = []

    cardinalities = list(train[categorical_cols].nunique().values+1)

    for name, cardinality in zip(categorical_cols, cardinalities):

        embedding_map.append({'cardinality': cardinality, 'size': min(50, (cardinality+1)//2)})



    # Define the neural network

    keras.backend.clear_session()

    inputs = [keras.Input(shape=(X_train[0].shape[1],))] + [keras.Input(shape=(1,)) for _ in categorical_cols]

    outputs = [inputs[0]]

    for cat_input, embedding, name in zip(inputs[1:], embedding_map, categorical_cols):

        out = Embedding(embedding['cardinality'], embedding['size'], input_length=1, name=name)(cat_input)

        outputs.append(Flatten()(out))   

    output = concatenate(outputs)

    output = Dense(1024, activation='relu')(output)

    output = Dropout(dropout)(output)

    output = Dense(512, activation='relu')(output)

    output = Dropout(dropout)(output)

    output = Dense(1, activation='sigmoid')(output)



    model = keras.Model(inputs=inputs, outputs=output)

    model.compile(

        loss='mean_absolute_error', 

        optimizer=keras.optimizers.Adam(learning_rate=learning_rate), 

        metrics=[get_rmspe(y_min, y_max)]

    )

    return model
def schedule(epoch):

    if epoch < 20:

        return 1e-3

    else:

        return 5e-4

    

callbacks = [keras.callbacks.LearningRateScheduler(schedule, verbose=0)]



epochs = 30

batch_size = 256

learning_rate = 2e-3

dropout = 0.05



y_max, y_min, X_train, y_train, X_val, y_val = get_data(has_validation=True)



model = get_model(y_min, y_max, learning_rate, dropout)

model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_val, y_val))
epochs = 30

batch_size = 256

learning_rate = 1e-3

dropout = 0.05



y_max, y_min, X_train, y_train = get_data(has_validation=False)

model = get_model(y_min, y_max, learning_rate, dropout)

model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, callbacks=callbacks)
test = test.set_index('Date')

X = test[all_cols]

ids = test.Id



# Now, convert the data into a list 

X_test = [X[continuous_cols].values] + [X[c].values[..., None] for c in categorical_cols]
y_pred = np.exp(model.predict(X_test) * (y_max - y_min) + y_min)
y_pred.min()
submit = pd.DataFrame({'Id': ids, 'Sales': y_pred.squeeze()}).sort_values('Id')

submit.to_csv('/kaggle/working/submission.csv', index=False)