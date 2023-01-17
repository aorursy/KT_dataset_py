import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.utils import to_categorical, normalize

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix
df_train = pd.read_csv('../input/glass.csv')

print(f'The train set contain {df_train.shape[0]} examples')

print(f'The train set contain {df_train.shape[1]} features')

df_train.head()
X_train = df_train.drop('Type', axis = 1)

y_train = df_train['Type']
glass_classes = y_train.unique()

values = y_train.value_counts()



plt.bar(glass_classes, values)

plt.title('Train set')

plt.xlabel('Glass Classes')

plt.ylabel('Examples count')

plt.show()
X_train.describe()
X_train = df_train.values

X_train = normalize(X_train)

print(X_train[0])
y_train = to_categorical(y_train)

y_train.shape
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.2)

X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, test_size = 0.5)
model = tf.keras.models.Sequential([

       

    tf.keras.layers.Dense(256, input_shape=(10,), activation='relu'),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Dropout(0.3),

    

    tf.keras.layers.Dense(256, activation='relu'),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Dropout(0.3),

    

    tf.keras.layers.Dense(512, activation='relu'),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Dropout(0.5),

            

    tf.keras.layers.Dense(8, activation='softmax')

])





model.compile(loss='categorical_crossentropy',

              optimizer=Adam(0.0001),

              metrics=['acc'])



model
model.summary()
history = model.fit(X_train, y_train,

                    epochs=400,

                    validation_data=(X_val, y_val),

                    verbose=2,

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
model.evaluate(X_test, y_test)
y_pred = model.predict(X_test)

y_pred_cl = np.argmax(y_pred, axis = 1)

y_true = np.argmax(y_test, axis = 1)



confusion_matrix(y_true, y_pred_cl)