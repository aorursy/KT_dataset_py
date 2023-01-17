import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib as mpl

import matplotlib.pyplot as plt

import datetime

import re

import os

#from mpl_toolkits.basemap import Basemap



print('*'*50)

print('Pandas Version    : ', pd.__version__)

print('Numpy Version     : ', np.__version__)

print('Matplotlib Version: ', mpl.__version__)

print('Seaborn Version   : ', sns.__version__)

print('*'*50)



sns.set_style('white')



pd.options.display.max_rows = 100

pd.options.display.max_columns = 100
from sklearn import metrics

import sklearn

from sklearn.model_selection  import train_test_split

from sklearn import linear_model

from sklearn import preprocessing

from sklearn.metrics import accuracy_score

from sklearn.naive_bayes import GaussianNB

from sklearn import svm

from sklearn import tree

from sklearn.neural_network import MLPClassifier
from __future__ import absolute_import, division, print_function, unicode_literals



import tensorflow as tf

from tensorflow.keras import layers



print(tf.version.VERSION)

print(tf.keras.__version__)



from tensorflow import keras

from keras.layers import Dense

from keras.utils import to_categorical

from keras.callbacks import Callback
data = pd.read_csv('../input/data.csv')

data.head()
data = data.loc[:,~data.columns.duplicated()]

data = data.loc[:, ~data.columns.str.contains('^Unnamed')]

#data.head()
toDrop = [

    'power_of_shot.1',

    'remaining_min.1',

    'remaining_sec.1',

    'distance_of_shot.1',

    'knockout_match.1',

    'date_of_game',

    'match_id',

    'team_id',

    'shot_basics',

    'range_of_shot',

    'match_event_id'

]

data.drop(toDrop, axis=1, inplace=True)

data.shape

#data.head()
data = data[np.isfinite(data['is_goal'])]

data.head()
mdata=data

toDrop = [

    'team_name',

    'home/away',

    'shot_id_number',

    #'lat/lng',

   # 'location_x',

  #  'location_y',

    'game_season'

]

data.drop(toDrop, axis=1, inplace=True)

data['Lat'], data['Lon'] = data['lat/lng'].str.split(', ').str

data.Lat=data.Lat.astype(float)

data.Lon=data.Lon.astype(float)

data.drop('lat/lng', axis=1, inplace=True)
data.shape
data=pd.get_dummies(data, columns=["area_of_shot"], prefix=["area"])
data.shape
data.info()
data.remaining_min.isnull().sum()

data['remaining_min']=data['remaining_min'].replace(np.NaN, int(data['remaining_min'].mean()))

data['power_of_shot']=data['power_of_shot'].replace(np.NaN, int(data['power_of_shot'].mean()))

data['knockout_match']=data['knockout_match'].replace(np.NaN, int(data['knockout_match'].mean()))

data['remaining_sec']=data['remaining_sec'].replace(np.NaN, int(data['remaining_sec'].mean()))

data['distance_of_shot']=data['distance_of_shot'].replace(np.NaN, int(data['distance_of_shot'].mean()))



data['location_x']=data['location_x'].replace(np.NaN, int(data['location_x'].mean()))

data['location_y']=data['location_y'].replace(np.NaN, int(data['location_y'].mean()))

data['Lat']=data['Lat'].replace(np.NaN, int(data['Lat'].mean()))

data['Lon']=data['Lon'].replace(np.NaN, int(data['Lon'].mean()))



data['is_goal']=data['is_goal'].replace(np.NaN, int(data['is_goal'].mean()))

data['type_of_shot']=data['type_of_shot'].replace(np.NaN, "shot - NaN")

data['type_of_combined_shot']=data['type_of_combined_shot'].replace(np.NaN, "shot - NaN")
data['remaining_min'] = data['remaining_min'].astype(int)

data['power_of_shot'] = data['power_of_shot'].astype(int)

data['knockout_match'] = data['knockout_match'].astype(int)

data['remaining_sec'] = data['remaining_sec'].astype(int)

data['is_goal'] = data['is_goal'].astype(int)



data['location_x'] = data['location_x'].astype(int)

data['location_y'] = data['location_y'].astype(int)



data['distance_of_shot'] = data['distance_of_shot'].astype(int)

#pd.get_dummies(data, columns=["area_of_shot"], prefix=["area"]).head()
data.tail()
#data['type_of_shot']=data['type_of_shot'].replace("shot - NaN","shot - 0" )

#data['Ignore1'], data['Shot_type'] = data['type_of_shot'].str.split(' - ').str

#data.drop(['Ignore1','type_of_shot'], axis=1, inplace=True)
#data['type_of_combined_shot']=data['type_of_combined_shot'].replace("shot - NaN","shot - 0" )

#data['Ignore2'], data['Shot_Combined_type'] = data['type_of_combined_shot'].str.split(' - ').str

#data.drop(['Ignore2','type_of_combined_shot'], axis=1, inplace=True)
#data
data=pd.get_dummies(data, columns=["type_of_shot"], prefix=["type_of_shot"])

data=pd.get_dummies(data, columns=["type_of_combined_shot"], prefix=["type_of_combined_shot"])

data
data.shape
y=data['is_goal']

x=data

x.drop('is_goal', axis=1, inplace=True)

print(y)
#x
#x.info()

#x.head()
data['location_x']=data['location_x']/250

data['location_y']=data['location_y']/350

data
x.shape
y.shape
#x_train, x_test = train_test_split(x, test_size=0.1)

#y_train, y_test = train_test_split(y, test_size=0.1)
#print(x_train.shape)

#print(x_test.shape)

#print(y_train.shape)

#print(y_test.shape)
y_binary = to_categorical(y)
y_binary.dtype
X_train, X_test, Y_train, Y_test = train_test_split(x, y_binary, test_size = 0.1, random_state = 15)
X_train2, X_test2, Y_train2, Y_test2 = train_test_split(x, y, test_size = 0.1, random_state = 15)
lm = linear_model.SGDClassifier(max_iter=20000)

lm.fit(X_train2, Y_train2)

Y_pred = lm.predict(X_test2)

mse = sklearn.metrics.mean_squared_error(Y_test2, Y_pred)

print("Mean Squared Error of SGDC Classifier: ",mse,"\n")
print (metrics.classification_report(Y_test2, Y_pred))
accuracy = accuracy_score(Y_pred, Y_test2)

print("Accuracy of SGDC Classifier: ",accuracy,"\n")
print(Y_pred.shape)

print(Y_test.shape)
clf_test = GaussianNB() 

clf_test = clf_test.fit(X_train2, Y_train2)

pred5 = clf_test.predict(X_test2)

print (metrics.classification_report(Y_test2,pred5))
accuracy = accuracy_score(pred5, Y_test2)

print("Accuracy of Gaussian Classifier: ",accuracy,"\n")
lm = MLPClassifier(hidden_layer_sizes=(100,50), max_iter=10000)

lm.fit(X_train2, Y_train2)

Y_pred2 = lm.predict(X_test2)

mse = sklearn.metrics.mean_squared_error(Y_test2, Y_pred2)

print("Mean Squared Error of MLP Classifier: ",mse,"\n")
accuracy = accuracy_score(Y_pred2, Y_test2)

print("Accuracy of MLP Classifier: ",accuracy,"\n")
lm = MLPClassifier(hidden_layer_sizes=(200,150,20,10), max_iter=10000)

lm.fit(X_train2, Y_train2)

Y_pred3 = lm.predict(X_test2)

mse = sklearn.metrics.mean_squared_error(Y_test2, Y_pred3)

print("Mean Squared Error of MLP Classifier: ",mse,"\n")
accuracy = accuracy_score(Y_pred3, Y_test2)

print("Accuracy of MLP Classifier: ",accuracy,"\n")
clf_test2 = tree.DecisionTreeClassifier(min_samples_split = 11) 

clf_test2 = clf_test2.fit(X_train2, Y_train2)

pred6 = clf_test2.predict(X_test2)

print (metrics.classification_report(Y_test2, pred6))
accuracy = accuracy_score(pred6, Y_test2)

print("Accuracy of MLP Classifier: ",accuracy,"\n")
clf_test3 = svm.SVC(kernel = 'rbf', C = 1.0) 

clf_test3 = clf_test3.fit(X_train2, Y_train2)

pred7 = clf_test3.predict(X_test2)

print (metrics.classification_report(Y_test2, pred7))
accuracy = accuracy_score(pred7, Y_test2)

print("Accuracy of MLP Classifier: ",accuracy,"\n")
X_train.shape
Y_train.shape
model = tf.keras.Sequential([

# Adds a densely-connected layer with 64 units to the model:

layers.Dense(10, activation='relu', input_shape=(80,)), # layer-10, 17, 80 for no oneshot ###lighten it up, what to wake

# Add another:

layers.Dense(20, activation='relu'),#layer-20

#layers.Dense(50, activation='relu'),

# Add a softmax layer with 10 output units:

layers.Dense(2, activation='softmax')])



#model.compile(optimizer=tf.train.AdamOptimizer(0.001),

#              loss='categorical_crossentropy',#categorical_crossentropy can be used only for binary

#              metrics=['accuracy'])

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
reducer = keras.callbacks.ReduceLROnPlateau(monitor="val_loss")
earlystopper = keras.callbacks.EarlyStopping(monitor='val_loss')
#dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train))

#dataset = dataset.batch(200).repeat()



#val_dataset = tf.data.Dataset.from_tensor_slices((X_test, Y_test))

#val_dataset = val_dataset.batch(200).repeat()



model.fit(X_train, Y_train, epochs=100, batch_size=30, validation_split=0.3, callbacks=[reducer]) #, validation_split=0.3) #)#, steps_per_epoch=30,call back #32 lr scheduler lr on plateau early stoop

         #validation_data=(X_test, Y_test)) #,

        #  validation_steps=3)
model.evaluate(X_test, Y_test, batch_size=30)
#result = model.predict((X_test, Y_test), batch_size=32)

result = model.predict(X_test, batch_size=32)

result

print(result.shape)
print(result)

result_goal = result[:,1]
print(result_goal)
result_goal = np.where(result_goal > 0.5, 1., 0.)
print(result_goal)

print(Y_test[:,1])
print(result_goal.shape)

print(Y_test[:,1].shape)
accuracy = accuracy_score(result_goal, Y_test[:,1])

print(accuracy)