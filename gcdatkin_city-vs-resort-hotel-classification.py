import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.express as px



from sklearn.preprocessing import LabelEncoder, StandardScaler

from sklearn.model_selection import train_test_split



import tensorflow as tf
data = pd.read_csv('../input/hotel-booking-demand/hotel_bookings.csv')
data
data.info()
data.isna().sum()
data = data.drop('company', axis=1)
for column in ['children', 'agent']:

    data[column] = data[column].fillna(data[column].mean())
def get_categorical_uniques(df):

    return {column: list(df[column].unique()) for column in df.columns if df.dtypes[column] == 'object'}
get_categorical_uniques(data)
data['reservation_year'] = data['reservation_status_date'].apply(lambda x: np.int(x[0:4]))

data['reservation_month'] = data['reservation_status_date'].apply(lambda x: np.int(x[5:7]))



data = data.drop('reservation_status_date', axis=1)
data
get_categorical_uniques(data)
data['meal'] = data['meal'].replace('Undefined', np.NaN)
target = 'hotel'





ordinal_features = ['arrival_date_month']



nominal_features = ['meal', 'country', 'market_segment', 'distribution_channel', 'reserved_room_type', 'assigned_room_type', 'deposit_type', 'customer_type', 'reservation_status']
month_ordering = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
data['arrival_date_month'] = data['arrival_date_month'].apply(lambda x: month_ordering.index(x))
numerical_columns = [column for column in data.columns if data.dtypes[column] != 'object']



corr = data[numerical_columns].corr()



plt.figure(figsize=(18, 15))

sns.heatmap(corr, annot=True, vmin=-1.0, vmax=1.0, cmap='mako')

plt.show()
data = data.drop(['arrival_date_week_number', 'reservation_year'], axis=1)
data
def onehot_encode(df, column, prefix):

    df = df.copy()

    dummies = pd.get_dummies(df[column], prefix=prefix)

    df = pd.concat([df, dummies], axis=1)

    df = df.drop(column, axis=1)

    return df
onehot_prefixes = ['m', 'c', 'ms', 'dc', 'rt', 'at', 'dt', 'ct', 'rs']
for column, prefix in zip(nominal_features, onehot_prefixes):

    data = onehot_encode(data, column, prefix)
data
label_encoder = LabelEncoder()



data['hotel'] = label_encoder.fit_transform(data['hotel'])
{index: label for index, label in enumerate(label_encoder.classes_)}
data
y = data.loc[:, target]

X = data.drop(target, axis=1)
scaler = StandardScaler()



X = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=46)
X.shape
y.mean()
inputs = tf.keras.Input(shape=(246,))

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

epochs = 7



history = model.fit(

    X_train,

    y_train,

    validation_split=0.2,

    batch_size=batch_size,

    epochs=epochs

)
fig = px.line(

    history.history,

    y=['loss', 'val_loss'],

    labels={'x': "Epoch", 'y': "Loss"},

    title="Loss Over Time"

)



fig.show()
np.argmin(history.history['val_loss'])
model.evaluate(X_test, y_test)