%matplotlib inline

import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (6,6)

import numpy as np
import pandas as pd

import os
print(os.listdir("../input"))
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")

X = np.array(train.drop(["label"], axis=1)) / 255.
y = np.array(train["label"])

from tensorflow.keras.utils import to_categorical
y = to_categorical(y)

X_test = np.array(test) / 255.

print(X.shape, y.shape, X_test.shape)
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15, random_state=42)
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_val = X_val.reshape(X_val.shape[0], 28, 28, 1)

X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

print(X_train.shape, X_val.shape, y_train.shape, y_val.shape, X_test.shape)
from tensorflow.keras.layers import Input, Dense, Dropout, Flatten, BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.models import Model
# https://github.com/keras-team/keras/issues/10306
# BatchNormalization makes autoencoder working better

inp = Input(shape=(28,28,1))

# Encoder
conv_1 = Conv2D(128, (3,3), padding="same", activation="relu")(inp)
conv_1 = BatchNormalization()(conv_1)
pool_1 = MaxPooling2D((2,2), padding="same")(conv_1)
conv_2 = Conv2D(32, (3,3), padding="same", activation="relu")(pool_1)
conv_2 = BatchNormalization()(conv_2)
pool_2 = MaxPooling2D((2,2), padding="same")(conv_2)
conv_3 = Conv2D(32, (3,3), padding="same", activation="relu")(pool_2)
conv_3 = BatchNormalization()(conv_3)
encoded = MaxPooling2D((2,2), padding="same")(conv_3)

# Decoder
conv_4 = Conv2D(32, (3,3), padding="same", activation="relu")(encoded)
up_1 = UpSampling2D((2,2))(conv_4)
conv_5 = Conv2D(32, (3,3), padding="same", activation="relu")(up_1)
up_2 = UpSampling2D((2,2))(conv_5)
conv_6 = Conv2D(128, (3,3), activation="relu")(up_2)
up_3 = UpSampling2D((2,2))(conv_6)
decoded = Conv2D(1, (3,3), activation='sigmoid', padding='same', name='autoencoder')(up_3)

# Classification
flatten = Flatten()(encoded)
fc = Dense(128, activation="relu")(flatten)
fc = Dropout(0.5)(fc)
classifer = Dense(y_train.shape[1], activation="softmax", name="classification")(fc)

model = Model(inp, [decoded, classifer])
losses = {
    "autoencoder": "mse",
    "classification": "categorical_crossentropy"
}

loss_weights = {
    "autoencoder": 1.0,
    "classification": 5.0
}

metrics = {
    "autoencoder": "mse",
    "classification": "acc"
}

model.compile(loss=losses, loss_weights=loss_weights, optimizer="adam", metrics=metrics)

model.summary()
from tensorflow.keras.callbacks import ModelCheckpoint

checkpoint = ModelCheckpoint('model.h5', 
                             monitor='val_classification_acc', 
                             verbose=1, 
                             save_best_only=True, 
                             mode='max', 
                             save_weights_only = True)

hist = model.fit(X_train,
                 {"autoencoder": X_train,  "classification": y_train},
                 verbose=1,
                 batch_size=512, 
                 epochs=50, 
                 validation_data=(X_val, {"autoencoder": X_val, "classification": y_val}), 
                 callbacks=[checkpoint])
y_1, y_2 = model.predict(X_val[:10])
y_1.shape, y_2.shape
samples = 5
y_1, y_2 = model.predict(X_val[:samples])
for i in range(samples):
    plt.subplot(1, 2, 1)
    plt.imshow(X_val[i].reshape(28,28), cmap="gray")
    plt.title("original")
    plt.subplot(1, 2, 2)
    plt.imshow(y_1[i].reshape(28,28), cmap="gray")
    plt.title("autoencoder and predict {}".format(y_2[i].argmax()))
    plt.show()
sub = pd.read_csv("../input/sample_submission.csv")
model.load_weights("model.h5")
y_test = model.predict(X_test, batch_size=1024, verbose=0)
sub.Label = np.argmax(y_test[1], axis=1)
sub.to_csv("submission.csv", index=False)
!head submission.csv
