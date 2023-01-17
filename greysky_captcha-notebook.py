from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten
from tensorflow.keras.optimizers import Adam

from io import BytesIO
from PIL import Image
import numpy as np
import glob
train_x, train_y = get_all_data("../input/captcha-dataset/dataset/*.png")
scaler = MinMaxScaler(feature_range=(0, 1))
encoder = LabelEncoder()

train_x = scaler.fit_transform(train_x)
train_y = encoder.fit_transform(train_y)
train_x, train_y = shuffle(train_x, train_y)
example = 32

plt.imshow(train_x[example].reshape(50, 36))
print("label: " + encoder.classes_[train_y[example]])
model = Sequential([
    Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same', input_shape=(50, 36, 1)),
    MaxPool2D(pool_size=(2, 2), strides=2),
    Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'),
    MaxPool2D(pool_size=(2, 2), strides=2),
    Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same'),
    MaxPool2D(pool_size=(2, 2), strides=2),
    Conv2D(filters=256, kernel_size=(3, 3), activation='relu', padding='same'),
    MaxPool2D(pool_size=(2, 2), strides=2),
    Flatten(),
    Dense(units=1024, activation="relu"),
    Dense(units=512, activation="relu"),
    Dense(units=34, activation='softmax')])

model.compile(optimizer=Adam(learning_rate=0.0001), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(train_x.reshape(-1, 50, 36, 1), train_y, batch_size=10, epochs=20)
captcha = Image.open("../input/captcha-dataset/dataset/N5BV6.png")

x = get_data(captcha, include_labels=False)
x = MinMaxScaler(feature_range=(0, 1)).fit_transform(x).reshape(-1, 50, 36, 1)

predictions = np.argmax(model.predict(x, batch_size=10), axis=-1)
captcha_label = "".join(encoder.classes_[predictions])

print("captcha label : N5BV6");
print("prediction : ", captcha_label)
def get_all_data(path):
    train_x = []
    train_y = []

    for filename in glob.glob(path):
        image = Image.open(filename)
        X, y = get_data(image, include_labels=True)
        train_x.append(X)
        train_y.append(y)

    return (np.vstack(train_x), np.hstack(train_y))
def get_data(image, include_labels=False):
    samples = []
    for i in range(5):
        samples.append(np.array(image.crop((36 * i, 0, 36 * (i + 1), 50))).reshape(-1,))
        
    if include_labels:
        return (np.vstack(samples), np.hstack(list(image.filename[-9:-4])))
    else:
        return np.vstack(samples)
