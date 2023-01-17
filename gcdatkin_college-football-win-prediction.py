import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



import re

from sklearn.preprocessing import MinMaxScaler

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split



import tensorflow as tf
data = pd.read_csv('../input/college-football-attendance-2000-to-2018/CFBeattendance.csv', encoding='latin-1')
data
features_to_drop = ['Date', 'Site', 'Team', 'Opponent']



data.drop(features_to_drop, axis=1, inplace=True)
data.isna().sum()
data.dtypes
categorical_features = ['Time', 'Rank', 'TV', 'Opponent_Rank', 'Conference']
def get_uniques(df, columns):

    return {column: list(df[column].unique()) for column in columns}
get_uniques(data, categorical_features)
binary_features = ['TV', 'New Coach', 'Tailgating']



ordinal_features = ['Time', 'Rank', 'Opponent_Rank']



nominal_features = ['Conference']
data['TV'].value_counts()
data['TV'] = data['TV'].apply(lambda x: 0 if x == 'Not on TV' else 1)
data['New Coach'] = data['New Coach'].astype(np.int)

data['Tailgating'] = data['Tailgating'].astype(np.int)
data
data['Rank'].unique()
data['Rank'] = data['Rank'].apply(lambda x: 26 if x == 'NR' else np.int(x))

data['Opponent_Rank'] = data['Opponent_Rank'].apply(lambda x: 26 if x == 'NR' else np.int(x))
time_ordering = sorted(data['Time'].unique())
data['Time'] = data['Time'].apply(lambda x: time_ordering.index(x))
data
data['Conference'].unique()
def onehot_encode(df, column):

    dummies = pd.get_dummies(df[column])

    df = pd.concat([df, dummies], axis=1)

    df.drop(column, axis=1, inplace=True)

    return df
data = onehot_encode(data, 'Conference')
data.drop([4355, 5442, 5449, 5456], axis=0, inplace=True)
y = data['Result']

X = data.drop('Result', axis=1)
y
y = y.apply(lambda x :re.search(r'^[^\s]*', x).group(0))
y.unique()
y[(y == 'NC') | (y == 'White') | (y == 'Blue')]
label_encoder = LabelEncoder()



y = label_encoder.fit_transform(y)

y_mappings = {index: value for index, value in enumerate(label_encoder.classes_)}

y_mappings
y
X
scaler = MinMaxScaler()



X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

X
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7)
inputs = tf.keras.Input(shape=(33,))

x = tf.keras.layers.Dense(16, activation='relu')(inputs)

x = tf.keras.layers.Dense(16, activation='relu')(x)

outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)



model = tf.keras.Model(inputs=inputs, outputs=outputs)





optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

metrics = [

    tf.keras.metrics.BinaryAccuracy(name='acc'),

    tf.keras.metrics.AUC(name='auc')

]



model.compile(

    optimizer=optimizer,

    loss='binary_crossentropy',

    metrics=metrics

)





batch_size = 32

epochs = 10



history = model.fit(

    X_train,

    y_train,

    validation_split=0.2,

    batch_size=batch_size,

    epochs=epochs,

    verbose=0

)
plt.figure(figsize=(14, 10))



epochs_range = range(1, epochs + 1)

train_loss = history.history['loss']

val_loss = history.history['val_loss']



plt.plot(epochs_range, train_loss, label="Training Loss")

plt.plot(epochs_range, val_loss, label="Validation Loss")



plt.title("Training and Validation Loss")

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.legend()



plt.show()
np.argmin(val_loss)
model.evaluate(X_test, y_test)
y.sum() / len(y)