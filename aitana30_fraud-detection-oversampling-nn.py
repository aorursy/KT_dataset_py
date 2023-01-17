# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
import keras

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#Creating the DataFrame
dataset = pd.read_csv("/kaggle/input/creditcardfraud/creditcard.csv")
#Let's take a look at the data we have available
dataset.head()
#Firstly, we will drop the "Time" column
dataset.drop(columns="Time", inplace=True)
#Let's analyze how many frauds and no-frauds there are in our data
dataset["Class"].value_counts()
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
Xy = scaler.fit_transform(dataset.values)
X = Xy[:, :-1]
y = Xy[:, -1]
# Creation of the train and test datasets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
from imblearn.over_sampling import RandomOverSampler
cc = RandomOverSampler(random_state=0)
X_resampled, y_resampled = cc.fit_resample(X_train, y_train)
from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder()
enc.fit(y_train.reshape(-1,1))
y_resampled = enc.transform(y_resampled.reshape(-1,1)).toarray()
y_test = enc.transform(y_test.reshape(-1,1)).toarray()

from kerastuner.tuners import RandomSearch
from tensorflow.keras import layers, Sequential, optimizers

def build_model(hp):
    model = Sequential()
    for i in range(hp.Int('num_layers', 1, 2)):
        model.add(layers.Dense(units=hp.Int('units_' + str(i),
                                            min_value=10, #mínimo número de neuronas en la capa
                                            max_value=50, #máximo número de neuronas en la capa
                                            step=10),
                               activation='relu')),
    model.add(layers.Dense(2, activation='softmax'))
    opt = optimizers.Adam(learning_rate=0.005)
    model.compile(optimizer=opt,
        loss='categorical_crossentropy',
        metrics=['accuracy'])
    return model

#max_trials is the number of combinations that will be tested. It is important to remark 
#that if it's not the number of total possibilities it won't be guaranteed that the result 
#obtained is the best one for our NN, it will be the best of the combinations tested.
#This is used because testing all the possibilities will take a lot of time.
tuner = RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=5, 
    executions_per_trial=3,
    directory='my_dir',
    project_name='helloworld',
    overwrite=True)
history = tuner.search(X_resampled, y_resampled,
             epochs=5,
             validation_data=(X_test, y_test))
models = tuner.get_best_models(num_models=3)
tuner.get_best_hyperparameters(num_trials=1)[0].get_config()
pred = np.argmax(models[0].predict(X_test), axis=1)
real = np.argmax(y_test, axis=1)
from sklearn.metrics import confusion_matrix
confusion_matrix(real, pred)
model = keras.Sequential(
    [
        keras.layers.Dense(40, activation="relu", name="initial", input_shape=(29,)),
        keras.layers.Dense(30, activation="relu", name="hidden"),
        keras.layers.Dense(2, activation="softmax", name="output")  
    ]
)
opt = keras.optimizers.Adam(learning_rate=0.005)
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=[tf.keras.metrics.FalseNegatives()])
history = model.fit(x=X_resampled,
                  y=y_resampled,
                  epochs = 10,
                  validation_data=(X_test, y_test))
pred = np.argmax(model.predict(X_test), axis=1)
real = np.argmax(y_test, axis=1)
from sklearn.metrics import confusion_matrix
confusion_matrix(real, pred)