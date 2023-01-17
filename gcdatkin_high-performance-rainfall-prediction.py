import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import RobustScaler

from sklearn.model_selection import train_test_split



import tensorflow as tf



from sklearn.metrics import f1_score
data = pd.read_csv("../input/weather-dataset-rattle-package/weatherAUS.csv")
data
pd.set_option('display.max_columns', None)
data.drop('Date', axis=1, inplace=True)
data.isnull().sum()
data.dtypes
data['WindDir9am'].unique()
data['RainToday'] = data['RainToday'].fillna('No')
encoder = LabelEncoder()



label_encoder_columns = ['RainToday', 'RainTomorrow']



for column in label_encoder_columns:

    data[column] = encoder.fit_transform(data[column])
def add_column_prefixes(data, column, prefix):

    return data[column].apply(lambda x: prefix + str(x))
data['WindDir9am'] = add_column_prefixes(data, 'WindDir9am', "9_")

data['WindDir3pm'] = add_column_prefixes(data, 'WindDir3pm', "3_")
data
pd.get_dummies(data['WindGustDir'])
def onehot_encoder(data, columns):

    for column in columns:

        dummies = pd.get_dummies(data[column])

        data = pd.concat([data, dummies], axis=1)

        data.drop(column, axis=1, inplace=True)

    return data
categorical_features = ['Location', 'WindGustDir', 'WindDir9am', 'WindDir3pm']



data = onehot_encoder(data, categorical_features)
data
data.isnull().sum()
def impute_means(data, columns):

    for column in columns:

        data[column] = data[column].fillna(data[column].mean())
na_columns = ['MinTemp',

              'MaxTemp',

              'Rainfall',

              'Evaporation',

              'Sunshine',

              'WindGustSpeed',

              'WindSpeed9am',

              'WindSpeed3pm',

              'Humidity9am',

              'Humidity3pm',

              'Pressure9am',

              'Pressure3pm',

              'Cloud9am',

              'Cloud3pm',

              'Temp9am',

              'Temp3pm']



impute_means(data, na_columns)
y = data['RainTomorrow']

X = data.drop('RainTomorrow', axis=1)
scaler = RobustScaler()



X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
X
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)
inputs = tf.keras.Input(shape=(117,))

x = tf.keras.layers.Dense(16, activation='relu')(inputs)

x = tf.keras.layers.Dense(16, activation='relu')(x)

outputs = tf.keras.layers.Dense(2, activation='softmax')(x)



model = tf.keras.Model(inputs=inputs, outputs=outputs)
model.summary()
model.compile(

    optimizer='adam',

    loss='sparse_categorical_crossentropy',

    metrics=['accuracy']

)
EPOCHS = 6

BATCH_SIZE = 32
history = model.fit(

    X_train,

    y_train,

    validation_split=0.2,

    epochs=EPOCHS,

    batch_size=BATCH_SIZE,

    verbose=1

)
plt.figure(figsize=(14, 10))



plt.plot(range(EPOCHS), history.history['loss'], color='b')

plt.plot(range(EPOCHS), history.history['val_loss'], color='r')



plt.xlabel('Epoch')

plt.ylabel('Loss')



plt.show()
np.argmin(history.history['val_loss'])
print(f"Model Accuracy: {model.evaluate(X_test, y_test, verbose=0)[1]}")
y.sum() / len(y)
y_pred = model.predict(X_test)
y_pred
y_test
y_pred = list(map(lambda x: np.argmax(x), y_pred))
print(f"Model F1 Score: {f1_score(y_test, y_pred)}")