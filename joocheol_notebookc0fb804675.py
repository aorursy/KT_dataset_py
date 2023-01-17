# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import tensorflow as tf



from tensorflow import keras

from tensorflow.keras import layers

from tensorflow.keras.layers.experimental import preprocessing



print(tf.__version__)
df = pd.read_csv('/kaggle/input/kaggle-practice-competition/train.csv')

df
train_features = df.copy()

train_labels = train_features.pop('MPG')
train_features.pop('Id')
train_features
horsepower = np.array(train_features['Horsepower'])



horsepower_normalizer = preprocessing.Normalization(input_shape=[1,])

horsepower_normalizer.adapt(horsepower)
horsepower_model = tf.keras.Sequential([

    horsepower_normalizer,

    layers.Dense(units=1)

])



horsepower_model.summary()
horsepower_model.predict(horsepower[:10])
horsepower_model.compile(

    optimizer=tf.optimizers.Adam(learning_rate=0.1),

    loss='mean_absolute_error')
%%time

history = horsepower_model.fit(

    train_features['Horsepower'], train_labels,

    epochs=100,

    # suppress logging

    verbose=0,

    # Calculate validation results on 20% of the training data

    validation_split = 0.2)
df = pd.read_csv('/kaggle/input/kaggle-practice-competition/test.csv')

df
hist = pd.DataFrame(history.history)

hist['epoch'] = history.epoch

hist.tail()
y = horsepower_model.predict(df['Horsepower'])
y
df = pd.read_csv('/kaggle/input/kaggle-practice-competition/sample.csv')

df
df['Predicted'] = y
df
df.to_csv('submission.csv', index=False)