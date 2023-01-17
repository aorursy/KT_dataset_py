import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split



import tensorflow as tf
data = pd.read_csv("../input/chocolate-bar-ratings/flavors_of_cacao.csv")
data
plt.figure(figsize=(12, 10))

sns.heatmap(data.corr(), annot=True)

plt.show()
data.drop(['REF', 'Review\nDate'], axis=1, inplace=True)
data
data.isnull().sum()
data = data.dropna(axis=0)
data.dtypes
data.columns = ['Company', 'SpecificOrigin', 'CocoaPercent', 'Location', 'Rating', 'BeanType', 'BroadOrigin']
data
def removePercents(data):

    return data.apply(lambda x: float(x.strip('%')) / 100)
data['CocoaPercent'] = removePercents(data['CocoaPercent'])
len(data['SpecificOrigin'].unique())
categorical_features = ['Company', 'SpecificOrigin', 'Location', 'BeanType', 'BroadOrigin']
def onehot_encode(data, columns):

    for column in columns:

        dummies = pd.get_dummies(data[column])

        data = pd.concat([data, dummies], axis=1)

        data.drop(column, axis=1, inplace=True)

    return data
data = onehot_encode(data, categorical_features)
y = data['Rating']

X = data.drop('Rating', axis=1)
X
scaler = MinMaxScaler()



X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)
inputs = tf.keras.Input(shape=(1655,))

x = tf.keras.layers.Dense(16, activation='relu')(inputs)

x = tf.keras.layers.Dense(16, activation='relu')(x)

outputs = tf.keras.layers.Dense(1)(x)



model = tf.keras.Model(inputs=inputs, outputs=outputs)
optimizer = tf.keras.optimizers.RMSprop(0.001)



model.compile(

    optimizer=optimizer,

    loss='mse'

)
model.summary()
epochs = 10

batch_size = 32



history = model.fit(

    X_train,

    y_train,

    validation_split=0.2,

    epochs=epochs,

    batch_size=batch_size,

    verbose=1

)
plt.figure(figsize=(14, 10))



plt.plot(range(epochs), history.history['loss'], color='b')

plt.plot(range(epochs), history.history['val_loss'], color='r')



plt.xlabel('Epoch')

plt.ylabel('Loss')



plt.show()
np.argmin(history.history['val_loss'])
model.evaluate(X_test, y_test)