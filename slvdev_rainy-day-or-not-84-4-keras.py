import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.utils import to_categorical, normalize

from sklearn.utils import shuffle

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix
df = pd.read_csv('../input/weatherAUS.csv')

df.head()
df.count().sort_values()
df.drop(columns=['Sunshine','Evaporation','Cloud3pm','Cloud9am', 'RISK_MM'], axis=1, inplace=True)
df.dropna(inplace=True)

df.head()
df['Date'] = pd.to_datetime(df['Date'])

df.set_index('Date', inplace=True)

df.sort_index(inplace=True)

df.head()
df['MaxTemp'].rolling(365).mean().plot()
df['Location'].unique()
df['Location'] = df['Location'].astype('category').cat.codes

df['WindGustDir'] = df['WindGustDir'].astype('category').cat.codes

df['WindDir9am'] = df['WindDir9am'].astype('category').cat.codes

df['WindDir3pm'] = df['WindDir3pm'].astype('category').cat.codes

df['RainToday'] = df['RainToday'].astype('category').cat.codes

df['RainTomorrow'] = df['RainTomorrow'].astype('category').cat.codes
df.head()
df.reset_index(drop=True, inplace=True)

df.head()
df = shuffle(df)

df.head()
X = df.drop('RainTomorrow', axis=1)

y = df['RainTomorrow']
X.describe()
X = X.values

X = normalize(X)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.2)

X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, test_size = 0.5)
X_train.shape
model = tf.keras.models.Sequential([

       

    tf.keras.layers.Dense(128, input_shape=(17,), activation='relu'),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Dropout(0.5),

    

    tf.keras.layers.Dense(64, activation='relu'),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Dropout(0.5),

    

    tf.keras.layers.Dense(32, activation='relu'),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Dropout(0.5),



            

    tf.keras.layers.Dense(1, activation='sigmoid')

])





model.compile(loss='binary_crossentropy',

              optimizer=Adam(0.00001),

              metrics=['acc'])
model.summary()
history = model.fit(X_train, y_train,

                    epochs=10,

                    validation_data=(X_val, y_val),

                    verbose=1,

                   )
plt.plot(history.history['acc'])

plt.plot(history.history['val_acc'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()
plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()
loss, accuracy = model.evaluate(X_test, y_test)
acc = accuracy * 100

plt.bar(1, acc)

plt.text(0.92,45,f'{acc:.2f}%', fontsize=20)

plt.title('Accuracy')

plt.xticks([])

plt.ylabel('Percent')

plt.show()