import numpy as np

import pandas as pd

import plotly.express as px



from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split



import tensorflow as tf



from tensorflow_addons.metrics import RSquare
data = pd.read_csv('../input/food-choices/food_coded.csv')
data
data.isna().sum()
data[(data.isna().sum()[data.isna().sum() >= 10]).index]
data = data.drop('type_sports', axis=1)
numeric_nulls = [column for column in data.columns if data.dtypes[column] != 'object' and data.isna().sum()[column] != 0]
for column in numeric_nulls:

    data[column] = data[column].fillna(data[column].mean())
{column: list(data[column].unique()) for column in data.columns if data.isna().sum()[column] > 0}
data['GPA'] = data['GPA'].replace('Personal ', np.NaN)

data['GPA'] = data['GPA'].replace('Unknown', np.NaN)

data['GPA'] = data['GPA'].replace('3.79 bitch', '3.79')



data['GPA'] = data['GPA'].astype(np.float)



data['GPA'] = data['GPA'].fillna(data['GPA'].mean())
nonnumeric_nulls = [column for column in data.columns if data.dtypes[column] == 'object' and data.isna().sum()[column] != 0]

nonnumeric_nulls.remove('weight')



data = data.drop(nonnumeric_nulls, axis=1)
data
data.isna().sum()
data['weight'].unique()
data['weight'] = data['weight'].replace('Not sure, 240', '240')

data['weight'] = data['weight'].replace('144 lbs', '144')

data['weight'] = data['weight'].replace("I'm not answering this. ", np.NaN)



data = data.dropna(axis=0).reset_index(drop=True)
data.isna().sum().sum()
data = data.astype(np.float)
data
y = data.loc[:, 'weight']

X = data.drop('weight', axis=1)
scaler = StandardScaler()



X = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=22)
X.shape
inputs = tf.keras.Input(shape=(48,))

x = tf.keras.layers.Dense(256, activation='relu')(inputs)

x = tf.keras.layers.Dense(512, activation='relu')(x)

x = tf.keras.layers.Dense(512, activation='relu')(x)

x = tf.keras.layers.Dense(256, activation='relu')(x)

outputs = tf.keras.layers.Dense(1, activation='linear')(x)



model = tf.keras.Model(inputs, outputs)





model.compile(

    optimizer='adam',

    loss='mse'

)





batch_size = 32

epochs = 200



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
y_preds = np.squeeze(model.predict(X_test))
rsquare = RSquare()



rsquare.update_state(y_test, y_preds)
print("R^2 Score:", rsquare.result().numpy())