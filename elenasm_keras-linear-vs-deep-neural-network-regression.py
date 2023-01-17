

import numpy as np 

import pandas as pd 





import tensorflow as tf

from tensorflow import keras

from tensorflow.keras import layers

from tensorflow.keras.layers.experimental import preprocessing



print(tf.__version__)



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



db = pd.read_csv('/kaggle/input/yeh-concret-data/Concrete_Data_Yeh.csv')

db.head() # = dv
db.isnull().sum()
db.describe() 
train_db = db.sample(frac=0.8, random_state= 10)

test_db = db.drop(train_db.index)

train_features = train_db.copy()

test_features = test_db.copy()



train_labels = train_features.pop('csMPa')

test_labels = test_features.pop('csMPa')



print(train_labels.head(), test_labels.head(), train_features.head(), test_features.head())
train_features.describe().transpose()[['mean', 'std']]
normalizer = preprocessing.Normalization()

normalizer.adapt(np.array(train_features))

print(normalizer.mean.numpy())
print(train_features[:2])

print(normalizer(np.array(train_features[:2])).numpy())
linear_model = tf.keras.Sequential([

    normalizer,

    layers.Dense(units=1)

]) # it produces units=1 outputs for each example
linear_model.predict(train_features) 
linear_model.compile(

    optimizer=tf.optimizers.Adam(learning_rate=0.1),

    loss='mean_absolute_error')
history = linear_model.fit(

    train_features, train_labels, 

    epochs=100,

    verbose=0,

    # Calculate validation results on 20% of the training data

    validation_split = 0.2)
test_results = {}

test_results['linear_model'] = linear_model.evaluate(

    test_features, test_labels, verbose=0) 
test_results
db['csMPa'].describe()
model = keras.Sequential([

      normalizer,

      layers.Dense(100, activation='relu'),

      layers.Dense(100, activation='relu'),

      layers.Dense(1)

  ]) 



model.compile(loss='mean_absolute_error',optimizer=tf.keras.optimizers.Adam(0.001))
model.summary()
history = model.fit(

    train_features, train_labels,

    validation_split=0.2,

    verbose=0, epochs=100)
test_results['dnn_model'] = model.evaluate(test_features, test_labels, verbose=0)
test_results 