# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np 

import pandas as pd

import tensorflow as tf

import tensorflow.keras.layers as L

import matplotlib.pyplot as plt

import plotly.graph_objects as go

import plotly.express as px

from sklearn.model_selection import train_test_split
data = pd.read_csv('../input/age-gender-and-ethnicity-face-data-csv/age_gender.csv')



## Converting pixels into numpy array

data['pixels']=data['pixels'].apply(lambda x:  np.array(x.split(), dtype="float32"))



data.head()
print('Total rows: {}'.format(len(data)))

print('Total columns: {}'.format(len(data.columns)))
## normalizing pixels data

data['pixels'] = data['pixels'].apply(lambda x: x/255)



## calculating distributions

age_dist = data['age'].value_counts()

ethnicity_dist = data['ethnicity'].value_counts()

gender_dist = data['gender'].value_counts().rename(index={0:'Male',1:'Female'})



def ditribution_plot(x,y,name):

    fig = go.Figure([

        go.Bar(x=x, y=y)

    ])



    fig.update_layout(title_text=name)

    fig.show()
ditribution_plot(x=age_dist.index, y=age_dist.values, name='Age Distribution')
ditribution_plot(x=ethnicity_dist.index, y=ethnicity_dist.values, name='Ethnicity Distribution')
ditribution_plot(x=gender_dist.index, y=gender_dist.values, name='Gender Distribution')
X = np.array(data['pixels'].tolist())



## Converting pixels from 1D to 3D

X = X.reshape(X.shape[0],48,48,1)
plt.figure(figsize=(16,16))

for i in range(1500,1520):

    plt.subplot(5,5,(i%25)+1)

    plt.xticks([])

    plt.yticks([])

    plt.grid(False)

    plt.imshow(data['pixels'].iloc[i].reshape(48,48))

    plt.xlabel(

        "Age:"+str(data['age'].iloc[i])+

        "  Ethnicity:"+str(data['ethnicity'].iloc[i])+

        "  Gender:"+ str(data['gender'].iloc[i])

    )

plt.show()
y = data['gender']



X_train, X_test, y_train, y_test = train_test_split(

    X, y, test_size=0.22, random_state=37

)
model = tf.keras.Sequential([

    L.InputLayer(input_shape=(48,48,1)),

    L.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),

    L.BatchNormalization(),

    L.MaxPooling2D((2, 2)),

    L.Conv2D(64, (3, 3), activation='relu'),

    L.MaxPooling2D((2, 2)),

    L.Flatten(),

    L.Dense(64, activation='relu'),

    L.Dropout(rate=0.5),

    L.Dense(1, activation='sigmoid')

])



model.compile(optimizer='sgd',

              loss=tf.keras.losses.BinaryCrossentropy(),

              metrics=['accuracy'])





## Stop training when validation loss reach 0.2700

class myCallback(tf.keras.callbacks.Callback):

    def on_epoch_end(self, epoch, logs={}):

        if(logs.get('val_loss')<0.2700):

            print("\nReached 0.2700 val_loss so cancelling training!")

            self.model.stop_training = True

        

callback = myCallback()



model.summary()
history = model.fit(

    X_train, y_train, epochs=20, validation_split=0.1, batch_size=64, callbacks=[callback]

)
fig = px.line(

    history.history, y=['loss', 'val_loss'],

    labels={'index': 'epoch', 'value': 'loss'}, 

    title='Training History')

fig.show()
loss, acc = model.evaluate(X_test,y_test,verbose=0)

print('Test loss: {}'.format(loss))

print('Test Accuracy: {}'.format(acc))
y = data['ethnicity']



X_train, X_test, y_train, y_test = train_test_split(

    X, y, test_size=0.22, random_state=37

)
model = tf.keras.Sequential([

    L.InputLayer(input_shape=(48,48,1)),

    L.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),

    L.MaxPooling2D((2, 2)),

    L.Conv2D(64, (3, 3), activation='relu'),

    L.MaxPooling2D((2, 2)),

    L.Flatten(),

    L.Dense(64, activation='relu'),

    L.Dropout(rate=0.5),

    L.Dense(5)

])



model.compile(optimizer='rmsprop',

              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),

              metrics=['accuracy'])





## Stop training when validation accuracy reach 79%

class myCallback(tf.keras.callbacks.Callback):

    def on_epoch_end(self, epoch, logs={}):

        if(logs.get('val_accuracy')>0.790):

            print("\nReached 79% val_accuracy so cancelling training!")

            self.model.stop_training = True

        

callback = myCallback()





model.summary()
history = model.fit(

    X_train, y_train, epochs=16, validation_split=0.1, batch_size=64, callbacks=[callback]

)
fig = px.line(

    history.history, y=['loss', 'val_loss'],

    labels={'index': 'epoch', 'value': 'loss'}, 

    title='Training History')

fig.show()
loss, acc = model.evaluate(X_test,y_test,verbose=0)

print('Test loss: {}'.format(loss))

print('Test Accuracy: {}'.format(acc))
y = data['age']



X_train, X_test, y_train, y_test = train_test_split(

    X, y, test_size=0.22, random_state=37

)
model = tf.keras.Sequential([

    L.InputLayer(input_shape=(48,48,1)),

    L.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),

    L.BatchNormalization(),

    L.MaxPooling2D((2, 2)),

    L.Conv2D(64, (3, 3), activation='relu'),

    L.MaxPooling2D((2, 2)),

    L.Conv2D(128, (3, 3), activation='relu'),

    L.MaxPooling2D((2, 2)),

    L.Flatten(),

    L.Dense(64, activation='relu'),

    L.Dropout(rate=0.5),

    L.Dense(1, activation='relu')

])



sgd = tf.keras.optimizers.SGD(momentum=0.9)



model.compile(optimizer='adam',

              loss='mean_squared_error',

              metrics=['mae'])





## Stop training when validation loss reach 110

class myCallback(tf.keras.callbacks.Callback):

    def on_epoch_end(self, epoch, logs={}):

        if(logs.get('val_loss')<110):

            print("\nReached 110 val_loss so cancelling training!")

            self.model.stop_training = True

        

callback = myCallback()





model.summary()
history = model.fit(

    X_train, y_train, epochs=20, validation_split=0.1, batch_size=64, callbacks=[callback]

)
fig = px.line(

    history.history, y=['loss', 'val_loss'],

    labels={'index': 'epoch', 'value': 'loss'}, 

    title='Training History')

fig.show()
mse, mae = model.evaluate(X_test,y_test,verbose=0)

print('Test Mean squared error: {}'.format(mse))

print('Test Mean absolute error: {}'.format(mae))