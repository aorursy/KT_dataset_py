import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from pandas_profiling import ProfileReport



import tensorflow as tf



from tensorflow import keras

from tensorflow.keras import layers

from tensorflow.keras.layers.experimental import preprocessing



%matplotlib inline
df = pd.read_csv("../input/fuelconsumption/FuelConsumptionCo2.csv")

df.head()
profile = ProfileReport(df, title='EDA Profiling')
profile.to_widgets()
df.describe().transpose()
model_data = df.drop(['MODELYEAR', 'MODEL'], 1)
model_data.head()
model_data = model_data[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_CITY', 'FUELCONSUMPTION_HWY', 'FUELCONSUMPTION_COMB', 'FUELCONSUMPTION_COMB_MPG', 'CO2EMISSIONS']]

model_data.head()
train_dataset = model_data.sample(frac=0.8, random_state=0)

test_dataset = model_data.drop(train_dataset.index)



train_features = train_dataset.copy()

test_features = test_dataset.copy()



train_labels = train_features.pop('CO2EMISSIONS')

test_labels = test_features.pop('CO2EMISSIONS')
normalizer = preprocessing.Normalization()

normalizer.adapt(np.array(train_features))
reg_multivar = tf.keras.Sequential([

    normalizer,

    layers.Dense(units=1)

])



reg_multivar.compile(

    optimizer=tf.optimizers.Adam(learning_rate=0.1),

    loss='mean_absolute_error')
%%time

history = reg_multivar.fit(

    train_features, train_labels, 

    epochs=100,

    verbose=0,

    validation_split = 0.2)
plt.plot(history.history['loss'], label='Treino')

plt.plot(history.history['val_loss'], label='Teste')

plt.xlabel('Época')

plt.ylabel('Erro [MAE]')

plt.legend()

plt.grid(True)
test_results = {}



test_results['Reg. Linear Multivariada'] = reg_multivar.evaluate(

    test_features, test_labels, verbose=0)
reg_dnn = tf.keras.Sequential([

    normalizer,

    layers.Dense(64, activation='relu'),

    layers.Dropout(0.1),

    layers.Dense(64, activation='relu'),

    layers.Dense(1)

])



reg_dnn.compile(

    optimizer=tf.optimizers.Adam(learning_rate=0.001),

    loss='mean_absolute_error')
reg_dnn.summary()
%%time

history = reg_dnn.fit(

    train_features, train_labels, 

    epochs=100,

    verbose=0,

    validation_split = 0.2)
plt.plot(history.history['loss'], label='Treino')

plt.plot(history.history['val_loss'], label='Teste')

plt.xlabel('Época')

plt.ylabel('Erro [MAE]')

plt.legend()

plt.grid(True)
test_results['Deep Neural Network'] = reg_dnn.evaluate(

    test_features, test_labels, verbose=0)
pd.DataFrame(test_results, index=['Mean Absolute Error [CO2EMISSIONS]']).T