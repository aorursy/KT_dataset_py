import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # for plots
import graphviz # print tree
from sklearn import datasets, tree, model_selection

from __future__ import absolute_import, division, print_function
import tensorflow as tf
from tensorflow import keras

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
X = pd.read_csv("../input/train.csv", usecols=['field','age','type', 'harvest_month']).values # harvest year foi retirado devido a análise 2
y = pd.read_csv("../input/train.csv", usecols=['production']).values
#X.head() #nao e possivel usar com o .values
#y.head() #nao e possivel usar com o .values

traindata = pd.read_csv("../input/train.csv", usecols=['field','age','type','harvest_month'])
traindata.head()

dataselect = pd.read_csv("../input/train.csv", usecols=['field','age','type','harvest_year','harvest_month'])
dataselect.head()

data = traindata.values
dataselect = dataselect.values

field_min = 0
field_max = 27
fieldData = {}
for i in range(field_min,field_max+1):
    file = "../input/field-"+str(i)+".csv"
    #fieldData[i] = pd.read_csv(file, usecols=['temperature','dewpoint','windspeed','Soilwater_L1','Precipitation']).values
    fieldData[i] = pd.read_csv(file, usecols=['temperature','Soilwater_L1','Precipitation']).values

def getDataField(field, month, year):
    #todos os dados possuem tamanho 192, e vão de 01/2002 a 12/2017, o que significa que com base no mês e ano podemos calcular o índice de onde queremos os dados
    offsetano = int(year) - 2002
    offsetmes = int(month) #
    return fieldData[int(field)][((offsetano*12) + offsetmes)-1]


dataP = []
for i in range(len(data)):
    climatic = getDataField(dataselect[i][0], dataselect[i][4], dataselect[i][3]).tolist()
    el = data[i].tolist()
    el = el + climatic
    dataP.append(el)
#print(dataP[5:])
dataP = np.array(dataP)

print('done')
y = pd.read_csv("../input/train.csv", usecols=['production']).values
X = dataP
print(y.shape)
print(X.shape)
y = y.flatten()
#X_train, X_test, y_train, y_test = model_selection.train_test_split(X,y,test_size=0.2, random_state = 33)
X_train = X
y_train = y

X_test = X
y_test = y

print(X_train.shape, y_train.shape)
#criação do model
def build_model():
  model = keras.Sequential([
    keras.layers.Dense(7, activation=tf.nn.relu,
                       input_shape=(X_train.shape[1],)),
    keras.layers.Dense(64, activation=tf.nn.relu),
    keras.layers.Dense(64, activation=tf.nn.relu),
    keras.layers.Dense(1)
  ])
  optimizer = tf.train.RMSPropOptimizer(0.001)
  model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae'])
  return model
model = build_model()
model.summary()
class PrintDot(keras.callbacks.Callback): # para monitoramnento do progresso
  def on_epoch_end(self, epoch, logs):
    if epoch % 100 == 0: print('')
    print('.', end='')

# The patience parameter is the amount of epochs to check for improvement
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=300)

history = model.fit(X_train, y_train, epochs=5000,
                    validation_split=0.3, verbose=0,
                    callbacks=[early_stop, PrintDot()])
def plot_history(history):
  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Abs Error')
  plt.plot(history.epoch, np.array(history.history['mean_absolute_error']),
           label='Train Loss')
  plt.plot(history.epoch, np.array(history.history['val_mean_absolute_error']),
           label = 'Val loss')
  plt.legend()
plot_history(history)
[loss, mae] = model.evaluate(X_test, y_test, verbose=1)
print("Testing set Mean Abs Error: ", (mae))
print("Testing set Loss: ", (loss))
test_predictions = model.predict(X_test)
print(test_predictions[30:])
print(test_predictions.shape)
print(y_test.shape)
plt.scatter(y_test, test_predictions)
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.axis('equal')
plt.xlim(plt.xlim())
plt.ylim(plt.ylim())
_ = plt.plot([-100, 100], [-100, 100])
testdfRead = pd.read_csv('../input/test.csv')
testdf = testdfRead.values

testdata = pd.read_csv("../input/test.csv", usecols=['field','age','type','harvest_month'])
testdata.head()
testdata = testdata.values

dataTest = []
for i in range(len(testdata)):
    climatic = getDataField(testdf[i][1], testdf[i][5], testdf[i][4]).tolist()
    el = testdata[i].tolist()
    el = el + climatic
    dataTest.append(el)
dataTest = np.array(dataTest)
print(dataTest.shape)

test_predictions = model.predict(dataTest)
test_predictions = test_predictions.flatten().tolist()
for i in range(len(test_predictions)):
    if test_predictions[i] < 0:
        test_predictions[i] = 0
    if test_predictions[i] > 1:
        test_predictions[i] = 1
submissiondf = submissiondf = pd.DataFrame(pd.DataFrame({'Id':testdfRead['Id'].values.flatten().tolist(),'production':test_predictions}))
submissiondf.to_csv('fit.csv', index=False)
import os
print(os.listdir("../"))
