# imports

import pandas as pd

import numpy as np

from keras import backend as K

import tensorflow as tf

from sklearn.preprocessing import StandardScaler

from keras.models import Sequential

from keras.layers import Dense

from sklearn.metrics import f1_score

from imblearn.over_sampling import SMOTE

from sklearn.impute import SimpleImputer
# retrieve data

test = pd.read_csv('/kaggle/input/equipfails/equip_failures_test_set.csv', na_values='na')

train = pd.read_csv('/kaggle/input/equipfails/equip_failures_training_set.csv', na_values='na')



# handle nan in test

test = test.interpolate(axis=1, limit=10, limit_direction='both').fillna(0)

train = train.interpolate(axis=1, limit=10, limit_direction='both').fillna(0); # ';' to suppress output
train.columns
train.describe()
train.head()
# trim cols to measures and target

cols = [col for col in train.columns if 'measure' in col]

train = train[cols + ['target']]

test = test[cols]
# split into input (X) and output (Y) variables

X = train.drop(columns='target').values

Y = train['target'].values

X_test = test.values
# standardize the input feature

sc = StandardScaler()

scaler = sc.fit(X)

X = scaler.transform(X)

X_test = scaler.transform(X_test)
# oversampling the failures to overcome class imbalance

sm = SMOTE(random_state=42)

X, Y = sm.fit_resample(X, Y)

X_test
# def metrics

def recall_metric(y_true, y_pred):

        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))

        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))

        recall = true_positives / (possible_positives + K.epsilon())

        return recall



def precision_metric(y_true, y_pred):

        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))

        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))

        precision = true_positives / (predicted_positives + K.epsilon())

        return precision



def f1_metric(y_true, y_pred):

    precision = precision_metric(y_true, y_pred)

    recall = recall_metric(y_true, y_pred)

    return 2*((precision*recall)/(precision+recall+K.epsilon()))
# build classifier neural network

classifier = Sequential()



classifier.add(Dense(30, activation='relu', kernel_initializer='random_normal', input_dim=100)) # Input Layer

classifier.add(Dense(15, activation='relu', kernel_initializer='random_normal')) # Hidden Layer

classifier.add(Dense(1, activation='sigmoid', kernel_initializer='random_normal')) # Output Layer



classifier.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', f1_metric]);
# Fitting the data to the training dataset

# using mini-batch of 50

classifier.fit(X,Y, batch_size=50, epochs=20)
# predict on test data

y_pred = classifier.predict(X_test)
# format dataframe to fit submission specification and write csv

submit = pd.DataFrame(y_pred)

submit[0] = submit[0].astype(int)

submit.index += 1

submit.rename(columns={0: 'target'})



# write csv

submit.to_csv('submission.csv', index_label='id', header=['target'])
# code from https://www.kaggle.com/rtatman/download-a-csv-file-from-a-kernel to get csv for submission

# import the modules we'll need

from IPython.display import HTML

import base64



# function that takes in a dataframe and creates a text link to  

# download it (will only work for files < 2MB or so)

def create_download_link(df, title = "Download CSV file", filename = "data.csv"):  

    csv = df.to_csv()

    b64 = base64.b64encode(csv.encode())

    payload = b64.decode()

    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'

    html = html.format(payload=payload,title=title,filename=filename)

    return HTML(html)



create_download_link(submit)