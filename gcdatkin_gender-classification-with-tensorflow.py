import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split



import tensorflow as tf
data = pd.read_csv('../input/gender-classification/Transformed Data Set - Sheet1.csv')
data
data.info()
data.isna().sum()
{column: list(data[column].unique()) for column in data.columns}
def add_prefixes(df, column, prefix):

    return df[column].apply(lambda x: prefix + x)
data['Favorite Beverage'] = add_prefixes(data, 'Favorite Beverage', 'b_')

data['Favorite Soft Drink'] = add_prefixes(data, 'Favorite Soft Drink', 's_')
data
def onehot_encode(df, columns):

    for column in columns:

        dummies = pd.get_dummies(df[column])

        df = pd.concat([df, dummies], axis=1)

        df.drop(column, axis=1, inplace=True)

    return df
data = onehot_encode(data, ['Favorite Music Genre', 'Favorite Beverage', 'Favorite Soft Drink'])
data
color_ordering = list(data['Favorite Color'].unique())

color_ordering
data['Favorite Color'] = data['Favorite Color'].apply(lambda x: color_ordering.index(x))
label_encoder = LabelEncoder()

data['Gender'] = label_encoder.fit_transform(data['Gender'])

gender_mappings = {index: value for index, value in enumerate(label_encoder.classes_)}
gender_mappings
data
plt.figure(figsize=(12, 10))

sns.heatmap(data.corr(), annot=True, vmin=-1, vmax=1)

plt.show()
y = data['Gender']

X = data.drop('Gender', axis=1)
scaler = MinMaxScaler()

X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
X
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7)
y.sum() / len(y)
inputs = tf.keras.Input(shape=(18,))

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

epochs = 24



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
len(y_test)