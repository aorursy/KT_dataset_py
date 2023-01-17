# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import tensorflow as tf

from tensorflow import keras

import matplotlib.pyplot as plt

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')

test = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')
train.describe()
# Checking for null values

train.isnull().any()
print("The shape of the training set is :", train.shape)

print("The shape of the test set is :", test.shape)
x_train = train.drop(['label'], axis=1).to_numpy()

y_train = train['label'].to_numpy()
sns.countplot(y_train)
x_train = x_train/255.0

x_test = test/255.0
# Reshaping the input shapes to get it in the shape which the model expects to recieve later.

x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)

print(x_train.shape, y_train.shape)
def build_model(hp):  

  model = keras.Sequential([

    keras.layers.Conv2D(

        filters=hp.Int('conv_1_filter', min_value=32, max_value=128, step=16),

        kernel_size=hp.Choice('conv_1_kernel', values = [3,5]),

        activation='relu',

        input_shape=(28,28,1)

    ),

    keras.layers.Conv2D(

        filters=hp.Int('conv_2_filter', min_value=32, max_value=64, step=16),

        kernel_size=hp.Choice('conv_2_kernel', values = [3,5]),

        activation='relu'

    ),

    keras.layers.Flatten(),

    keras.layers.Dense(

        units=hp.Int('dense_1_units', min_value=32, max_value=128, step=16),

        activation='relu'

    ),

    keras.layers.Dense(10, activation='softmax')

  ])

  

  model.compile(optimizer=keras.optimizers.Adam(hp.Choice('learning_rate', values=[1e-2, 1e-3])),

              loss='sparse_categorical_crossentropy',

              metrics=['accuracy'])

  

  return model

from kerastuner import RandomSearch

from kerastuner.engine.hyperparameters import HyperParameters
tuner_search = RandomSearch(build_model, objective='val_accuracy', max_trials=5, directory='output', project_name="Mnist Digit Recognizer")
tuner_search.search(x_train, y_train, epochs=3, validation_split=0.1)
model = tuner_search.get_best_models(num_models=1)[0]
model.summary()
model.fit(x_train, y_train, epochs=10, validation_split=0.1, initial_epoch=3)
# reshaping the testing array as in a similar way as perfromed for training array

test_array = np.array(x_test)

test_array = test_array.reshape(test_array.shape[0], 28, 28, 1)

print(test_array.shape)
predictions = model.predict(test_array)
# Creating a submission.csv file

test_predictions = []



for i in predictions:

    test_predictions.append(np.argmax(i))
submission =  pd.DataFrame({

        "ImageId": x_test.index+1,

        "Label": test_predictions

    })



submission.to_csv('my_submission.csv', index=False)
