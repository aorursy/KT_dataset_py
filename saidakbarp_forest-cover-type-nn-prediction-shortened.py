# Short version of forest cover type NN prediction
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir("../input"))
from tensorflow.python.data import Dataset
import tensorflow as tf
from tensorflow import keras

import warnings
warnings.filterwarnings("ignore")

tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.1f}'.format
# data loading to dataframe
data = pd.read_csv("../input/covtype.csv", sep=",")
data.head()
# splitting the data
from sklearn.model_selection import train_test_split
x=data[data.columns[:data.shape[1]-1]] # all columns except the last are x variables
y=data[data.columns[data.shape[1]-1:]]-1 # the last column as y variable
x_train, x_test, y_train, y_test = train_test_split(x, y , train_size = 0.5, random_state =  14)
#this randomizes dataset and splits to train and test data
#batch_normalization
from sklearn.preprocessing import StandardScaler

# training
norm_tcolumns=x_train[x_train.columns[:10]] # only the first ten columns need normalization, the rest is binary
scaler = StandardScaler().fit(norm_tcolumns.values)
scaledf = scaler.transform(norm_tcolumns.values)
training_examples = pd.DataFrame(scaledf, index=norm_tcolumns.index, columns=norm_tcolumns.columns) # scaledf is converted from array to dataframe
x_train.update(training_examples)

# validation
norm_vcolumns=x_test[x_test.columns[:10]]
vscaled = scaler.transform(norm_vcolumns.values) # this scaler uses std and mean of training dataset
validation_examples = pd.DataFrame(vscaled, index=norm_vcolumns.index, columns=norm_vcolumns.columns)
x_test.update(validation_examples)
# model construction: forming the network layers using keras

model = keras.Sequential([
 keras.layers.Dense(1024, activation=tf.nn.relu, input_shape=(x_train.shape[1],)), # neurons with relu activation, first layer with input 
 keras.layers.Dropout(0.5), # dropout for reducing the overfitting problem
 keras.layers.Dense(512, activation=tf.nn.relu), # 2nd hidden layer
 keras.layers.Dropout(0.5),
 keras.layers.Dense(256, activation=tf.nn.relu), # 3rd hidden layer
 keras.layers.Dropout(0.5),
 keras.layers.Dense(7, activation=tf.nn.softmax)]) #  output layer with 7 categories

model.compile(loss='sparse_categorical_crossentropy', #this loss method is useful for multiple categories, otherwise our model does not work
 optimizer=tf.train.AdamOptimizer(learning_rate=0.0043, beta1=0.9), metrics=['accuracy'])
# train the model
history1 = model.fit(x_train, y_train, epochs = 300, batch_size = 2048, verbose=0, validation_data = (x_test, y_test))
print('training acc.:',history1.history['acc'][-1],'\n','test acc.:', (history1.history['val_acc'])[-1])
# plot the accuracy history
import matplotlib.pyplot as plt
def plot_history(history1):
 plt.figure()
 plt.xlabel('Epoch')
 plt.ylabel('Accuracy %')
 plt.plot(history1.epoch, np.array(history1.history['acc']),
 label='Train Accuracy')
 plt.plot(history1.epoch, np.array(history1.history['val_acc']),
 label = 'Val Accuracy')
 plt.legend()
 plt.ylim([0.5, 1])
plot_history(history1)