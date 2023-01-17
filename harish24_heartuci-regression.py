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
import pandas as pd

data = pd.read_csv(r'../input/heart-disease-uci/heart.csv')
data.head()
from sklearn.model_selection import train_test_split

x = data.drop('target', axis=1)

y = data['target']
train_x, test_x, train_y, test_y = train_test_split(x,
                                                    y,
                                                    train_size=0.8,
                                                    stratify=y)
train_x = train_x.to_numpy()

test_x = test_x.to_numpy()

train_y = train_y.to_numpy()

test_y = test_y.to_numpy()
train_x.shape
!pip install tensorflow-addons
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.layers import Dense, Dropout, Flatten, BatchNormalization
from tensorflow.keras.models import Sequential

model = Sequential([
    Dense(64, activation='relu', input_shape=(13, )),
    Dropout(0.2),
    BatchNormalization(),
    Dense(32, activation='relu'),
    Dropout(0.5),
    BatchNormalization(),
    Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=[
                  tf.keras.metrics.Recall(), 'accuracy',
                  tf.keras.metrics.Precision()
              ])
history = model.fit(train_x,
                    train_y,
                    epochs=23,
                    validation_data=(test_x, test_y))
import plotly.offline as plo
import plotly.express as px
import plotly.graph_objs as go

dat = [
    go.Scatter(x=history.epoch,
               y=history.history['loss'],
               mode='lines',
               name='Loss')
]

fig = go.Figure(data=dat)

fig.add_trace(
    go.Scatter(x=history.epoch,
               y=history.history['recall_1'],
               mode='lines',
               name='recall'))

fig.add_trace(
    go.Scatter(x=history.epoch,
               y=history.history['val_loss'],
               mode='lines',
               name='val_loss'))

fig.add_trace(
    go.Scatter(x=history.epoch,
               y=history.history['val_recall_1'],
               mode='lines',
               name='val_recall'))

fig.update_layout(title='Learning Curve',
                  xaxis_title='Epochs',
                  yaxis_title='Metrics and Loss',
                  template='plotly_dark')
data1 = [
    go.Scatter(x=history.epoch,
               y=history.history['accuracy'],
               mode='lines',
               name='accuracy')
]

fig1 = go.Figure(data=data1)

fig1.add_trace(
    go.Scatter(x=history.epoch,
               y=history.history['val_accuracy'],
               mode='lines',
               name='val_accuracy'))

fig1.update_layout(title='Accuracy Curve',
                   xaxis_title='epochs',
                   yaxis_title='accuracy',
                   template='simple_white')
data2 = [
    go.Scatter(x=history.epoch,
               y=history.history['precision_1'],
               mode='lines',
               name='precision')
]

fig2 = go.Figure(data=data2)

fig2.add_trace(
    go.Scatter(x=history.epoch,
               y=history.history['val_precision_1'],
               mode='lines',
               name='val_precision'))

fig2.update_layout(title='Precision Curve',
                   xaxis_title='epochs',
                   yaxis_title='precision',
                   template='ggplot2')
pred = model.predict_classes(test_x)

df = pd.DataFrame(pred)