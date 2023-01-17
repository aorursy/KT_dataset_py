import numpy as np

import pandas as pd

import tensorflow as tf

import matplotlib.pyplot as plt

from keras.models import Sequential

from keras.layers import Dense

from keras.layers import Dropout

from keras.wrappers.scikit_learn import KerasClassifier

from keras.constraints import maxnorm

from keras.optimizers import SGD

from sklearn.model_selection import cross_val_score

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import StratifiedKFold

from sklearn.preprocessing import MinMaxScaler

from sklearn.pipeline import Pipeline
# fix random seed for reproducibility

seed = 7

np.random.seed(seed)
data = pd.read_csv('../input/No-show-Issue-Comma-300k.csv')

print(data.head())
data.rename(columns = {'ApointmentData':'AppointmentData', 

                       'Alcoolism': 'Alchoholism', 

                       'HiperTension': 'Hypertension',

                       'Handcap': 'Handicap',

                      'Sms_Reminder': 'SmsReminder'}, inplace = True)
y = data.ix[:,'Status']

x = data.drop('Status', axis=1)



print(x.head())
# encode class values as integers

encoder = LabelEncoder()

encoder.fit(y)

y = encoder.transform(y)
GenderDummies = pd.get_dummies(x.Gender, prefix='Gender').iloc[:,1:]

WeekdayDummies = pd.get_dummies(x.DayOfTheWeek).iloc[:, 1:]
x = pd.concat([x, GenderDummies, WeekdayDummies], axis = 1)

DropCols = ['AppointmentRegistration', 'AppointmentData', 'Gender', 'DayOfTheWeek']

x = x.drop(DropCols, axis=1)

print(x.head())
x = np.array(x).astype(float)

y = np.array(y)



# normalize the dataset

scaler = MinMaxScaler(feature_range=(0, 1))

x = scaler.fit_transform(x)

print(x)
# Create

model = Sequential()

model.add(Dense(32, input_dim=x.shape[1], activation='relu'))

model.add(Dropout(0.2))

model.add(Dense(16, activation='relu'))

model.add(Dropout(0.2))

model.add(Dense(1, activation='sigmoid'))



# Compile

epochs = 5

learning_rate = 0.1

decay_rate = learning_rate / epochs

momentum = 0.8

sgd = SGD(lr=learning_rate, momentum=momentum, decay=decay_rate, nesterov=False)

model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])



# Fit the model

history = model.fit(x, y, validation_split=0.2, epochs=epochs, batch_size=20, verbose=2)                                                                                        
# Summarize history for accuracy

plt.plot(history.history['acc'])

plt.plot(history.history['val_acc'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')
# Summarize history for loss

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')