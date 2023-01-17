import numpy as np

import pandas as pd

import seaborn as sns

import datetime

import matplotlib.pyplot as plt

import keras

from keras import Model

from keras import metrics

from keras.models import *

from keras.layers import *

from keras.optimizers import *

from sklearn import model_selection

from sklearn.model_selection import train_test_split

import os



seed = 42
df = pd.read_csv('../input/renfe.csv')
df.shape
df.head()
#df.drop(columns = ['Unnamed: 0'])

df.describe()
df.isnull().sum()
df['price'].fillna(df['price'].mean(),inplace=True)
df.dropna(inplace=True)
df.head()
df['insert_date'] = pd.to_datetime(df['insert_date'])

df['start_date'] = pd.to_datetime(df['start_date'])

df['end_date'] = pd.to_datetime(df['end_date'])



df['start_date_weekday'] = df['start_date'].dt.weekday

df['start_date_month'] = df['start_date'].dt.month

df['start_date_hour'] = df['start_date'].dt.hour

df['start_date_day'] = df['start_date'].dt.day

df['start_date_minute'] = df['start_date'].dt.minute

df['end_date_weekday'] = df['end_date'].dt.weekday

df['end_date_month'] = df['end_date'].dt.month

df['end_date_hour'] = df['end_date'].dt.hour

df['end_date_day'] = df['end_date'].dt.day

df['end_date_minute'] = df['end_date'].dt.minute
df['duration_h'] = (df['end_date'] - df['start_date']).astype('timedelta64[h]')

df['duration_m'] = (df['end_date'] - df['start_date']).astype('timedelta64[m]')
df.head()
sns.countplot(df['duration_m'])
sns.distplot(df['price'])
sns.countplot(df.origin)
sns.countplot(df.destination)
sns.countplot(df.train_type)
sns.countplot(df.train_class)
sns.countplot(df.fare)
from sklearn.preprocessing import MinMaxScaler

scaler_y = MinMaxScaler()

y=np.reshape(df['price'].values, (-1,1))

print(scaler_y.fit(y))

yscale=scaler_y.transform(y)
df['price_norm'] = yscale
#stack input data

def prepare_data_for_nn(frame, categorical_vars, numerical_vars, text_vars, label_col) : 

    inputs = []

    for c in categorical_vars :     

        cat_values = np.asarray(frame[c].tolist())

        inputs.append(np.array(pd.factorize(cat_values)[0]))

    if numerical_vars:

        inputs.append(frame[numerical_vars].values)

    return inputs, frame[label_col]

categorical_vars = ['start_date_month',

                    'start_date_day',

                    'start_date_weekday',

                    'start_date_hour',

                    'duration_h',

                    'end_date_month',

                    'end_date_day',

                    'end_date_weekday',

                    'end_date_hour',

                    'fare',

                    'train_class',

                    'train_type',

                    'destination', 

                    'origin']

numerical_vars = []

text_vars = []



inpts, outpts = prepare_data_for_nn(

        df,

        categorical_vars, 

        numerical_vars, 

        text_vars, 

        'price')
def split_data(inputs, output, train_part=0.8, test_part=0.1, valid_part = 0.1):

    X_train = []

    X_val = []

    X_test = []

    for input_feature in inputs:

        i_train, i_valtest = train_test_split(input_feature, test_size=test_part + valid_part, random_state=seed)

        i_test, i_val = train_test_split(i_valtest, test_size=test_part/(test_part + valid_part), random_state=seed)

        X_train.append(i_train)

        X_test.append(i_test)

        X_val.append(i_val)



    y_train, y_valtest = train_test_split(output, test_size=test_part + valid_part, random_state=seed)

    y_test, y_val = train_test_split(y_valtest, test_size=test_part/(test_part + valid_part), random_state=seed)

    return X_train, X_val, X_test, y_train, y_val, y_test
def emb_model_train(xtrain, xval, ytrain, yval, balanced_df):

    cat_inputs = []

    cat_outputs = []

    for cat_column in categorical_vars:

        cat_size = balanced_df[cat_column].nunique()

        cat_input = Input(shape=(1,), name=cat_column + '_input')

        embedding_size = min(np.ceil((cat_size)/2), 50 )

        embedding_size = int(embedding_size)

        x = Embedding(cat_size+1, embedding_size, input_length=1)(cat_input)

        cat_output = Flatten()(x)

        cat_inputs.append(cat_input)

        cat_outputs.append(cat_output)

    concatenate_inputs = []

    concatenate_inputs.extend(cat_outputs)



    lyr = keras.layers.concatenate(concatenate_inputs)

    lyr = Dense(128, activation="relu")(lyr)

    lyr = Dropout(0.3)(lyr)    

    lyr = Dense(64, activation="relu")(lyr)

    lyr = Dropout(0.3)(lyr)

    main_output = Dense(1, activation='relu', name='main_output')(lyr)



    all_inputs = []

    all_inputs.extend(cat_inputs)



    t_model = Model(inputs= all_inputs, outputs=[main_output])



    #t_model.summary()



    t_model.compile(loss="mse",optimizer=Adam(),metrics=['mse', 'mae']

    )

    checkpointer = keras.callbacks.ModelCheckpoint(filepath="weights_best.hdf5", verbose=1, save_best_only=True)

    t_model.fit(

        xtrain, 

        ytrain,

        batch_size=16,

        epochs=5,

        shuffle=True,

        validation_data = (xval, yval),

        callbacks=[checkpointer],

        verbose=1)

    #results = t_model.evaluate(xtest, ytest)

    

    #y_pred = t_model.predict(xtest)

    #y_pred_bool = np.round(y_pred)

    

    #return t_model
x_train, x_test, x_val, y_train, y_test, y_val = split_data(inpts, outpts)
emb_model_train(x_train, x_val, y_train, y_val, df)

from keras.models import load_model

model = load_model('weights_best.hdf5')

#model.load_weights('weights_best.hdf5')

predicted = model.evaluate(x_test, y_test)
print("mean squared error on test sample is {}".format(predicted[1]))