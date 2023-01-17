# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import pandas as pd
import numpy as np
import tensorflow as tf
import keras
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/student-mat.csv")

data.head()
print("Null values check ", data.columns[data.isna().any()].tolist())
columns  = data.columns;
for coloumn in columns :
    if (not str(data[coloumn].dtype).startswith("int")):
            print("Coloum Name ",coloumn, " Type ", data[coloumn].dtype)
            print("Unique values for ", coloumn , data[coloumn].unique() , "\n")
            values = data[coloumn].unique()
            convertor = dict(zip(values,range(len(values))))
            data[coloumn] = [convertor[item] for item in data[coloumn]]
data.describe()
print('Data Size ', data.shape)
training_features = ['school', 'sex', 'age', 'address', 'famsize', 'Pstatus', 'Medu', 'Fedu','Mjob', 'Fjob', 'reason', 'guardian', 'traveltime', 'studytime',
       'failures', 'schoolsup', 'famsup', 'paid', 'activities', 'nursery',
       'higher', 'internet', 'romantic', 'famrel', 'freetime', 'goout', 'Dalc',
       'Walc', 'health', 'absences', 'G1', 'G2']
label_feature = ['G3']
selected_feature_data = data
m = data.shape[0]
n = len(training_features)
percentage_of_training = 80
number_example_in_training = int((percentage_of_training * m)/100)
number_example_in_test = int(m - number_example_in_training)

print('number_example_in_training', number_example_in_training)
print('number_example_in_test', number_example_in_test)

training_data_features = selected_feature_data.head(number_example_in_training)[training_features]
training_data_labels = selected_feature_data.head(number_example_in_training)[label_feature]

test_data_features = selected_feature_data.head(number_example_in_test)[training_features]
test_data_labels = selected_feature_data.head(number_example_in_test)[label_feature]


print('shape of traing data features', training_data_features.shape)
print('shape of traing data labels', training_data_labels.shape)
print('shape of test data features', test_data_features.shape)
print('shape of test data features', test_data_labels.shape)
def build_model():

  model = keras.models.Sequential([
    keras.layers.Dense(19, activation=tf.nn.relu,kernel_regularizer= keras.regularizers.l2(0.01),
                       input_shape=(training_data_features.shape[1],)),
    keras.layers.Dense(13, activation=tf.nn.relu, kernel_regularizer= keras.regularizers.l2(0.01)),  
    keras.layers.Dense(7, activation=tf.nn.relu,kernel_regularizer= keras.regularizers.l2(0.01)),
    keras.layers.Dense(1)
  ])

  optimizer = tf.train.RMSPropOptimizer(learning_rate=0.006)

  model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae'])
  return model

model = build_model()
model.summary()
class PrintDot(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    if epoch % 100 == 0: print('')
    print('.', end='')

EPOCHS = 1000

# Store training stats
history = model.fit(training_data_features, training_data_labels, epochs=EPOCHS,
                    validation_split=0.2, verbose=0,
                    callbacks=[PrintDot()])
import matplotlib.pyplot as plt


def plot_history(history):
  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Abs Error [1000$]')
  plt.plot(history.epoch, np.array(history.history['mean_absolute_error']),
           label='Train Loss')
  plt.plot(history.epoch, np.array(history.history['val_mean_absolute_error']),
           label = 'Val loss')
  plt.legend()
  plt.ylim([0, 5])

plot_history(history)
model.evaluate(test_data_features, test_data_labels, verbose=0)
from sklearn.metrics import mean_squared_error

predicted_value = model.predict(test_data_features)
MSE = mean_squared_error(np.asmatrix(test_data_labels), predicted_value)
print("Mean Square Error ", MSE)
for i in range(0,10):
    print(predicted_value[i], "--", np.asmatrix(test_data_labels)[i])