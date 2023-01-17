import numpy as np

import pandas as pd

from pandas_profiling import ProfileReport

import matplotlib.pyplot as plt



from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split



import tensorflow as tf

from tensorflow.keras import Input

from tensorflow.keras.layers import Dense

from tensorflow.keras import Model
data = pd.read_csv('../input/iris/Iris.csv')
data
data.drop('Id', axis=1, inplace=True)
profile = ProfileReport(data)
profile.to_notebook_iframe()
y = data['Species']

X = data.drop('Species', axis=1)
encoder = LabelEncoder()



y = encoder.fit_transform(y)

species_mappings = {index: label for index, label in enumerate(encoder.classes_)}



species_mappings
pd.DataFrame(X)
scaler = StandardScaler()

X = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7)
inputs = Input(shape=(4,))

x = Dense(16, activation='relu')(inputs)

outputs = Dense(3, activation='softmax')(x)



model = Model(inputs=inputs, outputs=outputs)
model.summary()

tf.keras.utils.plot_model(model)
model.compile(optimizer='adam',

              loss='sparse_categorical_crossentropy',

              metrics=['accuracy'])
batch_size = 32

epochs = 400
history = model.fit(X_train, y_train, validation_split=0.2, batch_size=batch_size, epochs=epochs)
plt.figure(figsize=(14, 10))

plt.plot(range(epochs), history.history['loss'], color='blue')

plt.plot(range(epochs), history.history['val_loss'], color='red')

plt.title("Learning Curves for Training/Validation Loss")

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.legend(['Training Loss', 'Validation Loss'])

plt.show()
model.evaluate(X_test, y_test)