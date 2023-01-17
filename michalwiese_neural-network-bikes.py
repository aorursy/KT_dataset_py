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
# Kaggle competition https://www.kaggle.com/c/bike-sharing-demand
import plotly.express as px

import plotly.graph_objects as go

import tensorflow as tf

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Dropout

from tensorflow.keras.callbacks import EarlyStopping
# Ładowanie i eksploracja danych
train_dataset = pd.read_csv("../input/bike-sharing-demand/train.csv")

test_dataset = pd.read_csv("../input/bike-sharing-demand/test.csv")



df = train_dataset.copy()

test_df = test_dataset.copy()
season=pd.get_dummies(df['season'],prefix='season')

df=pd.concat([df,season],axis=1)

season=pd.get_dummies(test_df['season'],prefix='season')

test_df=pd.concat([test_df,season],axis=1)





weather=pd.get_dummies(df['weather'],prefix='weather')

df=pd.concat([df,weather],axis=1)

weather=pd.get_dummies(test_df['weather'],prefix='weather')

test_df=pd.concat([test_df,weather],axis=1)





df.drop(['season','weather'],inplace=True,axis=1)

df.head()

test_df.drop(['season','weather'],inplace=True,axis=1)

test_df.head()
df.isnull().sum()
df.info()
test_df.info()
df.drop(['casual', 'registered'], inplace = True, axis = 1)

df.info()
px.histogram(df, x='count')
df['hour'] = [t.hour for t in pd.DatetimeIndex(df.datetime)]

df['day'] = [t.dayofweek for t in pd.DatetimeIndex(df.datetime)]

df['month'] = [t.month for t in pd.DatetimeIndex(df.datetime)]

df['year'] = [t.year for t in pd.DatetimeIndex(df.datetime)]

df['year'] = df['year'].map({2011 : 0,

                             2012 : 1})



test_df["hour"] = [t.hour for t in pd.DatetimeIndex(test_df.datetime)]

test_df["day"] = [t.dayofweek for t in pd.DatetimeIndex(test_df.datetime)]

test_df["month"] = [t.month for t in pd.DatetimeIndex(test_df.datetime)]

test_df['year'] = [t.year for t in pd.DatetimeIndex(test_df.datetime)]

test_df['year'] = test_df['year'].map({2011 : 0, 

                                       2012 : 1})
df.drop('datetime',axis=1,inplace=True)
from sklearn.model_selection import train_test_split,cross_validate



x_train, x_test, y_train, y_test = train_test_split(df.drop('count', axis=1), 

                                                    df['count'],

                                                    train_size = 0.9,

                                                    random_state=42)
x_train.head(2)
x_test.head(2)
print(x_train.shape)

print(x_test.shape)

print(y_train.shape)

print(y_test.shape)
x_train = x_train.values

x_test = x_test.values
# Budowa modelu
def build_model():

    model = Sequential()

    model.add(Dense(1024, kernel_regularizer = 'l2', activation = 'relu', input_shape=(1*18,)))

    model.add(Dense(512, activation='relu'))

    model.add(Dense(256, activation='relu'))

    model.add(Dense(128, activation='relu'))

    model.add(Dense(64, activation='relu'))

    model.add(Dense(1))



    model.compile(optimizer='adam',

                  loss='msle',

                  metrics=['mae', 'msle'])

    return model
model = build_model()

model.summary()
history = model.fit(x_train, y_train, epochs=150, validation_split=0.2, verbose=1, batch_size = 32)
def plot_hist(history):

    hist = pd.DataFrame(history.history)

    hist['epoch'] = history.epoch

    hist['rmsle'] = np.sqrt(hist['msle'])

    hist['val_rmsle'] = np.sqrt(hist['val_msle'])



    fig = go.Figure()

    fig.add_trace(go.Scatter(x=hist['epoch'], y=hist['rmsle'], name='RMSLE', mode='markers+lines'))

    fig.add_trace(go.Scatter(x=hist['epoch'], y=hist['val_rmsle'], name='val_RMSLE', mode='markers+lines'))

    fig.update_layout(width=1000, height=500, title="RMSLE vs. val_RMSLE", xaxis_title='Epoki', yaxis_title='Root Mean Squared Logarithmic Error')

    fig.show()



plot_hist(history)
# Sprawdzanie modelu
#Sprawdzenie metryk na zbiorze testowym



for name, value in zip(model.metrics_names, model.evaluate(x_test, y_test)):

    print(f'{name:8}{value:.4f}')
pred = model.predict(x_test)
prediction = pd.DataFrame(y_test)

prediction['pred'] = pred

prediction.head()
fig = px.scatter(prediction, 'count', 'pred')

fig.add_trace(go.Scatter(x=[0, 1000], y=[0, 1000], mode='lines'))

fig.show()
prediction['error'] = prediction['count'] - prediction['pred']

prediction.head()
px.histogram(prediction, 'error', marginal='rug', width=1000)
dt = test_df.drop(['datetime'],axis=1)

test_pred = model.predict(dt)



test_prediction = pd.DataFrame(test_pred)
test_prediction.columns = ['count']
test_prediction.head()
datetime = test_dataset['datetime']
datetime2 = pd.DataFrame(datetime)
datetime2.columns = ['datetime']

datetime2.head()
df_results = pd.concat([datetime2, test_prediction], axis=1)

df_results.head()
df_results.to_csv('DL_MW.csv', index=False)