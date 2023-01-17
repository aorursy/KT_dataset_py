import pandas as pd

from sklearn.model_selection import train_test_split

import tensorflow as tf

import matplotlib.pyplot as plt
df = pd.read_csv("/kaggle/input/dont-overfit-ii/train.csv", index_col="id")
df.info()
X = df.drop(columns=["target"])

y = df.filter(["target"])



X_train, X_test, y_train, y_test = train_test_split(

    X.values, y.values, test_size=0.2, random_state=42

)
X_train.shape, X_test.shape, y_train.shape, y_test.shape
input_layer = tf.keras.Input(shape=(300,))



hidden_layer_1 = tf.keras.layers.Dense(1024, activation='relu')(input_layer)



hidden_layer_2 = tf.keras.layers.Dense(512, activation='relu')(hidden_layer_1)



hidden_layer_3 = tf.keras.layers.Dense(256, activation='relu')(hidden_layer_2)



output_layer = tf.keras.layers.Dense(1, activation='sigmoid')(hidden_layer_3)



model = tf.keras.Model(inputs=input_layer, outputs=output_layer)
model.summary()
model.compile(

    optimizer='adam',

    loss='binary_crossentropy',

    metrics=['accuracy']

)



history = model.fit(

    X_train, y_train,

    batch_size=64,

    epochs=100,

    validation_split=0.25

)
plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('Loss')

plt.legend(['Train', 'Validation'], loc='upper left')

plt.show()



plt.plot(history.history['accuracy'])

plt.plot(history.history['val_accuracy'])

plt.title('Accuracy')

plt.legend(['Train', 'Validation'], loc='upper left')

plt.show()
input_layer = tf.keras.Input(shape=(300,))



hidden_layer_1 = tf.keras.layers.Dense(32, activation='relu')(input_layer)  # Simplifying model: less layers, less neurons

hidden_layer_1 = tf.keras.layers.Dropout(rate=0.2)(hidden_layer_1)  # Adding dropout layers



hidden_layer_2 = tf.keras.layers.Dense(16, activation='relu')(hidden_layer_1)  # Simplifying model: less layers, less neurons

hidden_layer_2 = tf.keras.layers.Dropout(rate=0.2)(hidden_layer_2)  # Adding dropout layers



output_layer = tf.keras.layers.Dense(1, activation='sigmoid')(hidden_layer_2)



model = tf.keras.Model(inputs=input_layer, outputs=output_layer)



model.compile(

    optimizer='adam',

    loss='binary_crossentropy',

    metrics=['accuracy']

)



es_callback = tf.keras.callbacks.EarlyStopping(

    monitor='val_loss',

    patience=3

)



history = model.fit(

    X_train, y_train,

    batch_size=64,

    epochs=100,

    callbacks=[es_callback],  # Stop training process earlier

    validation_split=0.25

)
plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('Loss')

plt.legend(['Train', 'Validation'], loc='upper left')

plt.show()



plt.plot(history.history['accuracy'])

plt.plot(history.history['val_accuracy'])

plt.title('Accuracy')

plt.legend(['Train', 'Validation'], loc='upper left')

plt.show()