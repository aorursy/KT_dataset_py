import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.express as px



from sklearn.preprocessing import LabelEncoder, StandardScaler

from sklearn.model_selection import train_test_split



import tensorflow as tf
data = pd.read_csv('../input/white-wine-quality/winequality-white.csv', delimiter=';')
data
corr = data.corr()



plt.figure(figsize=(12, 10))

sns.heatmap(corr, annot=True, vmin=-1.0, vmax=1.0)

plt.show()
data.info()
print("Total null values:", data.isna().sum().sum())
data['quality'].unique()
encoder = LabelEncoder()



data['quality'] = encoder.fit_transform(data['quality'])

{index: label for index, label in enumerate(encoder.classes_)}
y = data['quality']

X = data.drop('quality', axis=1)
scaler = StandardScaler()



X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
X
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=34)
num_features = X.shape[1]

print(num_features)
num_classes = len(y.unique())

print(num_classes)
inputs = tf.keras.Input(shape=(num_features,))

x = tf.keras.layers.Dense(64, activation='relu')(inputs)

x = tf.keras.layers.Dense(64, activation='relu')(x)

outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)



model = tf.keras.Model(inputs=inputs, outputs=outputs)





model.compile(

    optimizer='adam',

    loss='sparse_categorical_crossentropy',

    metrics=['accuracy']

)





batch_size = 32

epochs = 100



history = model.fit(

    X_train,

    y_train,

    validation_split=0.2,

    batch_size=batch_size,

    epochs=epochs,

    callbacks=[tf.keras.callbacks.ReduceLROnPlateau()]

)
fig = px.line(

    history.history,

    y=['loss', 'val_loss'],

    labels={'x': "Epoch", 'y': "Loss"},

    title="Loss Over Time"

)



fig.show()
model.evaluate(X_test, y_test)
data['quality'].value_counts()
pd.qcut(data['quality'], q=2, labels=[0,1]).value_counts()
y = pd.qcut(data['quality'], q=2, labels=[0,1])

X = data.drop('quality', axis=1)
scaler = StandardScaler()



X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=34)
inputs = tf.keras.Input(shape=(num_features,))

x = tf.keras.layers.Dense(64, activation='relu')(inputs)

x = tf.keras.layers.Dense(64, activation='relu')(x)

outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)



model = tf.keras.Model(inputs=inputs, outputs=outputs)





model.compile(

    optimizer='adam',

    loss='binary_crossentropy',

    metrics=['accuracy']

)





batch_size = 32

epochs = 100



history = model.fit(

    X_train,

    y_train,

    validation_split=0.2,

    batch_size=batch_size,

    epochs=epochs,

    callbacks=[tf.keras.callbacks.ReduceLROnPlateau()]

)
fig = px.line(

    history.history,

    y=['loss', 'val_loss'],

    labels={'x': "Epoch", 'y': "Loss"},

    title="Loss Over Time"

)



fig.show()
model.evaluate(X_test, y_test)