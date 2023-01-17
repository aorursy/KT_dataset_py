import numpy as np

import pandas as pd

import plotly.express as px



from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split



import tensorflow as tf
data = pd.read_csv('../input/womens-international-football-results/results.csv')
data
data.info()
data['year'] = data['date'].apply(lambda x: x[0:4])

data['month'] = data['date'].apply(lambda x: x[5:7])



data = data.drop('date', axis=1)
data
data['home_victory'] = (data['home_score'] > data['away_score']).astype(np.int)



data = data.drop(['home_score', 'away_score'], axis=1)
data['neutral'] = data['neutral'].astype(np.int)
data
def onehot_encode(df, columns, prefixes):

    df = df.copy()

    for column, prefix in zip(columns, prefixes):

        dummies = pd.get_dummies(df[column], prefix=prefix)

        df = pd.concat([df, dummies], axis=1)

        df = df.drop(column, axis=1)

    return df
data = onehot_encode(

    data,

    ['home_team', 'away_team', 'tournament', 'city', 'country'],

    ['home', 'away', 'tourn', 'city', 'country']

)
data
y = data.loc[:, 'home_victory']

X = data.drop('home_victory', axis=1)
scaler = StandardScaler()



X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=67)
X.shape
y.mean()
inputs = tf.keras.Input(shape=(1502,))

x = tf.keras.layers.Dense(64, activation='relu')(inputs)

x = tf.keras.layers.Dense(64, activation='relu')(x)

outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)



model = tf.keras.Model(inputs, outputs)





model.compile(

    optimizer='adam',

    loss='binary_crossentropy',

    metrics=[

        'accuracy',

        tf.keras.metrics.AUC(name='auc')

    ]

)





batch_size = 32

epochs = 20



history = model.fit(

    X_train,

    y_train,

    validation_split=0.2,

    batch_size=batch_size,

    epochs=epochs,

    callbacks=[tf.keras.callbacks.ReduceLROnPlateau()]

)
np.argmax(history.history['val_auc'])
fig = px.line(

    history.history,

    y=['loss', 'val_loss'],

    labels={'x': "Epoch", 'y': "Loss"},

    title="Loss Over Time"

)



fig.show()
fig = px.line(

    history.history,

    y=['auc', 'val_auc'],

    labels={'x': "Epoch", 'y': "AUC"},

    title="AUC Over Time"

)



fig.show()
model.evaluate(X_test, y_test)