%matplotlib inline
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import keras
from keras import metrics
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
import tensorflow as tf

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
df = pd.read_csv('/kaggle/input/pima-indians-diabetes-database/diabetes.csv')
df
df.describe()
df = pd.read_csv('/kaggle/input/pima-indians-diabetes-database/diabetes.csv')
target_column = ['Outcome']
predictors = list(set(list(df.columns))-set(target_column))
df[predictors] = df[predictors] / df[predictors].max()

X = df[predictors].values
y = df[target_column].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 0)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state = 0)
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(8, input_dim=8, activation = 'relu'))
model.add(tf.keras.layers.Dense(1, activation = 'sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['accuracy'])
history = model.fit(X_train,y_train,validation_data=(X_val, y_val), epochs=150, batch_size=3, verbose = 0)
print(history.history.keys())
_, accuracy = model.evaluate(X_train, y_train)
_, validation = model.evaluate(X_val, y_val)
print('Train accuracy: %.2f' % (accuracy))
print('Test accuracy: %.2f' % (validation))
loss_train = history.history['val_loss']
loss_val = history.history['loss']
epochs = range(1,151)
plt.plot(epochs, loss_train, 'g', label='Training loss')
plt.plot(epochs, loss_val, 'b', label='validation loss')
plt.title('Training and Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

loss_train = history.history['accuracy']
loss_val = history.history['val_accuracy']
epochs = range(1,151)
plt.plot(epochs, loss_train, 'g', label='Training accuracy')
plt.plot(epochs, loss_val, 'b', label='validation accuracy')
plt.title('Training and Validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
def create_model(learn_rate = 0.01, neurons = 3):
    model = tf.keras.Sequential()
    model.add(tf.keras.Dense(neurons, input_dim=8, activation = 'relu'))
    model.add(tf.keras.Dense(1, activation = 'sigmoid'))
    optimizer = keras.optimizers.Adam(lr=learn_rate)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

model = KerasClassifier(build_fn=create_model, verbose = 0)

batch_size = [3, 5, 10, 20]
epochs = [50, 100, 150, 200, 250]
learn_rate = [0.001, 0.01, 0.1]
neurons = [6, 8, 12]

param_grid = dict(learn_rate = learn_rate, batch_size=batch_size, epochs=epochs, neurons = neurons )

grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs = 1, cv = 3)
grid_result = grid.fit(X_train ,y_train)

print('Best: %f using %s' %(grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev,param in zip(means, stds, params):
    print('%f, (%f) with %r' % (mean, stdev, param))
# Full code

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import keras
from keras import metrics
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier

target_column = ['Outcome']
predictors = list(set(list(df.columns))-set(target_column))
df[predictors] = df[predictors] / df[predictors].max()

X = df[predictors].values
y = df[target_column].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 0)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state = 0)

print(X_train.shape)
print(X_test.shape)

model = tf.keras.Sequential()

model.add(tf.keras.layers.Dense(8, input_dim=8, activation = 'relu'))
model.add(tf.keras.layers.Dense(1, activation = 'sigmoid'))

#default learning rate is 0.001
model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['accuracy'])

history = model.fit(X_train,y_train,validation_data=(X_val, y_val), epochs=250, batch_size=10, verbose = 0)

print(history.history.keys())

_, accuracy = model.evaluate(X_train, y_train)
_, validation = model.evaluate(X_val, y_val)
print('Train accuracy: %.2f' % (accuracy))
print('Test accuracy: %.2f' % (validation))

loss_train = history.history['val_loss']
loss_val = history.history['loss']
epochs = range(1,251)
plt.plot(epochs, loss_train, 'g', label='Training loss')
plt.plot(epochs, loss_val, 'b', label='validation loss')
plt.title('Training and Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

loss_train = history.history['accuracy']
loss_val = history.history['val_accuracy']
epochs = range(1,251)
plt.plot(epochs, loss_train, 'g', label='Training accuracy')
plt.plot(epochs, loss_val, 'b', label='validation accuracy')
plt.title('Training and Validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()