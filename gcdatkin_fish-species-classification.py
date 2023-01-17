import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split



from sklearn.linear_model import LogisticRegression

import tensorflow as tf
data = pd.read_csv('../input/fish-market/Fish.csv')
data
data.isnull().sum()
y = data['Species']

X = data.drop('Species', axis=1)
X
scaler = StandardScaler()

X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
X
encoder = LabelEncoder()

y = encoder.fit_transform(y)

y_mappings = {index: label for index, label in enumerate(encoder.classes_)}
y_mappings
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7)
logistic_model = LogisticRegression()

logistic_model.fit(X_train, y_train)



logistic_model.score(X_test, y_test)
inputs = tf.keras.Input(shape=(6,))

x = tf.keras.layers.Dense(16, activation='relu')(inputs)

x = tf.keras.layers.Dense(16, activation='relu')(x)

outputs = tf.keras.layers.Dense(7, activation='softmax')(x)



nn_model = tf.keras.Model(inputs=inputs, outputs=outputs)





nn_model.compile(

    optimizer='adam',

    loss='sparse_categorical_crossentropy',

    metrics=['accuracy']

)





batch_size = 32

epochs = 1000



history = nn_model.fit(

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



plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.legend()



plt.show()
np.argmin(val_loss)
nn_model.evaluate(X_test, y_test)