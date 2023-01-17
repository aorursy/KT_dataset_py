from keras.datasets import cifar10

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

num_classes = 10
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
X_train = X_train.astype("float32") / 255
X_test = X_test.astype("float32") / 255

print("X_train shape:", X_train.shape)
print(X_train.shape[0], "train samples")
print(X_test.shape[0], "test samples")

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
model = keras.Sequential(
    [
        keras.Input(shape=X_train[0].shape),
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(len(y_test[0]), activation="softmax"),
    ]
)
    
model.summary()
batch_size = 128
epochs = 20
learning_rate=0.001

zoo = [
       (keras.optimizers.SGD(learning_rate=0.001), "SGD"),
       (keras.optimizers.Adam(learning_rate=0.001), "Adam"),
       (keras.optimizers.Adagrad(learning_rate=0.001), "Adagrad")
      ]
results = {}
for pair in zoo:
    opt = keras.optimizers.SGD(learning_rate=0.001)
    model.compile(loss="categorical_crossentropy", optimizer=pair[0], metrics=["accuracy"])
    model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.15)
    score = model.evaluate(X_test, y_test, verbose=0)
    results[pair[1]] = (score[0], score[1])
for result in results:
    print(result)
    print("Test loss:", results[result][0])
    print("Test accuracy:", results[result][1])