import numpy as np

import pandas as pd

import plotly.express as px



from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split



import tensorflow as tf
data = pd.read_csv('../input/covid19-healthy-diet-dataset/Food_Supply_Quantity_kg_Data.csv')
data
data.info()
data = data.drop('Unit (all except Population)', axis=1)
data.isna().sum()
for column in data.columns:

    if data.dtypes[column] != 'object' and data.isna().sum()[column] > 0:

        data[column] = data[column].fillna(data[column].mean())
data['Undernourished'].value_counts()
undernourished_numeric = data.loc[data['Undernourished'] != '<2.5', 'Undernourished'].astype(np.float)

undernourished_numeric
undernourished_numeric = undernourished_numeric.fillna(undernourished_numeric.mean())

undernourished_numeric = pd.qcut(undernourished_numeric, q=3, labels=[1, 2, 3])

undernourished_numeric
data.loc[undernourished_numeric.index, 'Undernourished'] = undernourished_numeric
data['Undernourished'] = data['Undernourished'].apply(lambda x: 0 if x == '<2.5' else x)
data['Undernourished'].value_counts()
data
data = data.drop('Country', axis=1)



data = data.drop(['Confirmed', 'Recovered', 'Active'], axis=1)
pd.qcut(data['Deaths'], q=2, labels=[0, 1]).value_counts()
data['Deaths'] = pd.qcut(data['Deaths'], q=2, labels=[0, 1])
data
y = data['Deaths']

X = data.drop('Deaths', axis=1)
scaler = StandardScaler()



X = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)
X.shape
inputs = tf.keras.Input(shape=(26,))

x = tf.keras.layers.Dense(64, activation='relu')(inputs)

x = tf.keras.layers.Dense(64, activation='relu')(x)

outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)



model = tf.keras.Model(inputs=inputs, outputs=outputs)





model.compile(

    optimizer='adam',

    loss='binary_crossentropy',

    metrics=[

        'accuracy',

        tf.keras.metrics.AUC(name='auc')

    ]

)





batch_size = 64

epochs = 14



history = model.fit(

    X_train,

    y_train,

    validation_split=0.2,

    batch_size=batch_size,

    epochs=epochs,

    verbose=0

)
fig = px.line(

    history.history,

    y=['loss', 'val_loss'],

    labels={'index': "Epoch", 'value': "Loss"},

    title="Training and Validation Loss"

)



fig.show()
np.argmin(history.history['val_loss'])
model.evaluate(X_test, y_test)
len(y_test)