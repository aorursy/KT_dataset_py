import tensorflow as tf
from tensorflow import keras
from sklearn.datasets import make_classification


X,y =make_classification(n_samples=5000, n_informative = 8, n_features=10, n_redundant = 0, n_classes=5, random_state=1)
print(X.shape, y.shape)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)   #Split data into 70% training, 30%test data
from sklearn.preprocessing import StandardScaler    #Standardize data
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
y_train = tf.keras.utils.to_categorical(y_train)
y_test = tf.keras.utils.to_categorical(y_test)
tf.keras.backend.clear_session()
model = tf.keras.Sequential()
model.add(tf.keras.Input(shape=10,))
model.add(tf.keras.layers.Dense(units=30, activation='relu', )) #First hidden layer with 30 neurons, relu activation
model.add(tf.keras.layers.Dense(units=15, activation='relu')) #Second hidden layer with 15 neurons, relu activation
model.add(tf.keras.layers.Dense(units=5, activation='softmax')) #Output layer with 5 neuron, softmax activation
model.summary()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
keras_history = model.fit(X_train, y_train , batch_size=16, epochs=200, validation_data=(X_test, y_test))
keras_history.history.keys()
import matplotlib.pyplot as plt
plt.plot(keras_history.history['loss'], label='Training loss')
plt.plot(keras_history.history['val_loss'], color='red', label='Test loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
tf.keras.backend.clear_session()
model = tf.keras.Sequential()
model.add(tf.keras.Input(shape=10,))
model.add(tf.keras.layers.Dense(units=30, activation='relu', activity_regularizer=tf.keras.regularizers.l2(0.01))) 
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Dropout(0.125))
model.add(tf.keras.layers.Dense(units=15, activation='relu', activity_regularizer=tf.keras.regularizers.l2(0.01))) 
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Dropout(0.25))
model.add(tf.keras.layers.Dense(units=5, activation='softmax')) 
model.summary()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
keras_history = model.fit(X_train, y_train , batch_size=16, epochs=200, validation_data=(X_test, y_test))
import matplotlib.pyplot as plt
plt.plot(keras_history.history['loss'], label='Training loss')
plt.plot(keras_history.history['val_loss'], color='red', label='Test loss')
plt.legend()