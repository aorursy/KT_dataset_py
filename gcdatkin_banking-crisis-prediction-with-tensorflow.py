import numpy as np

import pandas as pd

import plotly.express as px



from sklearn.preprocessing import LabelEncoder, StandardScaler

from sklearn.model_selection import train_test_split



import tensorflow as tf
data = pd.read_csv('../input/africa-economic-banking-and-systemic-crisis-data/african_crises.csv')
data
((data['banking_crisis'] == 'crisis').astype(int) == data['systemic_crisis']).all()
data.isna().sum()
data = data.drop(['case', 'country'], axis=1)
data
cc3_dummies = pd.get_dummies(data['cc3'])

data = pd.concat([data, cc3_dummies], axis=1)

data = data.drop('cc3', axis=1)
data
y = data['banking_crisis']

X = data.drop('banking_crisis', axis=1)
y
label_encoder = LabelEncoder()



y = label_encoder.fit_transform(y)

{index: label for index, label in enumerate(label_encoder.classes_)}
y = pd.Series(y).apply(lambda x: 1 - x)
scaler = StandardScaler()



X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
X
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7)
X.shape
y.sum() / len(y)
inputs = tf.keras.Input(shape=(23,))

x = tf.keras.layers.Dense(64, activation='relu')(inputs)

x = tf.keras.layers.Dense(64, activation='relu')(x)

outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)



model = tf.keras.Model(inputs=inputs, outputs=outputs)





model.compile(

    optimizer='adam',

    loss='binary_crossentropy',

    metrics=[tf.keras.metrics.AUC(name="auc")]

)





batch_size = 64

epochs = 60



history = model.fit(

    X_train,

    y_train,

    validation_split=0.2,

    batch_size=batch_size,

    epochs=epochs,

    callbacks=[tf.keras.callbacks.ReduceLROnPlateau()],

    verbose=0

)
fig = px.line(

    history.history,

    y=['loss', 'val_loss'],

    labels={'index': "Epoch", 'value': "Loss"},

    title="Training and Validation Loss"

)



fig.show()
model.evaluate(X_test, y_test)