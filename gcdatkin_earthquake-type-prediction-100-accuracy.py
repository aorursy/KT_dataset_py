import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split



import tensorflow as tf
data = pd.read_csv('../input/earthquake-database/database.csv')
data
data.info()
data = data.drop('ID', axis=1)
data.isna().sum()
null_columns = data.loc[:, data.isna().sum() > 0.66 * data.shape[0]].columns
data = data.drop(null_columns, axis=1)
data.isna().sum()
data['Root Mean Square'] = data['Root Mean Square'].fillna(data['Root Mean Square'].mean())
data = data.dropna(axis=0).reset_index(drop=True)
data.isna().sum().sum()
data
data['Month'] = data['Date'].apply(lambda x: x[0:2])

data['Year'] = data['Date'].apply(lambda x: x[-4:])



data = data.drop('Date', axis=1)
data['Month'] = data['Month'].astype(np.int)
data[data['Year'].str.contains('Z')]
invalid_year_indices = data[data['Year'].str.contains('Z')].index



data = data.drop(invalid_year_indices, axis=0).reset_index(drop=True)
data['Year'] = data['Year'].astype(np.int)
data['Hour'] = data['Time'].apply(lambda x: np.int(x[0:2]))



data = data.drop('Time', axis=1)
data
data['Status'].unique()
data['Status'] = data['Status'].apply(lambda x: 1 if x == 'Reviewed' else 0)
numeric_columns = [column for column in data.columns if data.dtypes[column] != 'object']
corr = data[numeric_columns].corr()
plt.figure(figsize=(12, 10))

sns.heatmap(corr, annot=True, vmin=-1.0, vmax=1.0)

plt.show()
numeric_columns.remove('Status')
scaler = StandardScaler()

standardized_df = pd.DataFrame(scaler.fit_transform(data[numeric_columns].copy()), columns=numeric_columns)
plt.figure(figsize=(18, 10))

for column in numeric_columns:

    sns.kdeplot(standardized_df[column], shade=True)

plt.xlim(-3, 3)

plt.show()
data
data['Type'].unique()
def onehot_encode(df, columns, prefixes):

    df = df.copy()

    for column, prefix in zip(columns, prefixes):

        dummies = pd.get_dummies(df[column], prefix=prefix)

        df = pd.concat([df, dummies], axis=1)

        df = df.drop(column, axis=1)

    return df
data = onehot_encode(

    data,

    ['Type', 'Magnitude Type', 'Source', 'Location Source', 'Magnitude Source'],

    ['t', 'mt', 's', 'ls', 'ms']

)
data
y = data.loc[:, 'Status']

X = data.drop('Status', axis=1)
scaler = StandardScaler()



X = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=56)
X.shape
y.mean()
inputs = tf.keras.Input(shape=(104,))

x = tf.keras.layers.Dense(64, activation='relu')(inputs)

x = tf.keras.layers.Dense(64, activation='relu')(x)

outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)



model = tf.keras.Model(inputs, outputs)





model.compile(

    optimizer='adam',

    loss='binary_crossentropy',

    metrics=[tf.keras.metrics.AUC(name='auc')]

)





batch_size = 32

epochs = 30



history = model.fit(

    X_train,

    y_train,

    validation_split=0.2,

    batch_size=batch_size,

    epochs=epochs,

    callbacks=[tf.keras.callbacks.ReduceLROnPlateau()],

    verbose=0

)
plt.figure(figsize=(18, 6))



epochs_range = range(epochs)

train_loss, val_loss = history.history['loss'], history.history['val_loss']

train_auc, val_auc = history.history['auc'], history.history['val_auc']



plt.subplot(1, 2, 1)

plt.plot(epochs_range, train_loss, label="Training Loss")

plt.plot(epochs_range, val_loss, label="Validation Loss")

plt.legend()

plt.title("Loss Over Time")



plt.subplot(1, 2, 2)

plt.plot(epochs_range, train_auc, label="Training AUC")

plt.plot(epochs_range, val_auc, label="Validation AUC")

plt.legend()

plt.title("AUC Over Time")



plt.show()
model.evaluate(X_test, y_test)
len(y_test)