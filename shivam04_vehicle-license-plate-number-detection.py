import pandas as pd
import urllib
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import os
import time
from PIL import Image

from keras.applications.vgg16 import VGG16
from keras.layers import Flatten, Dense, Conv2D, MaxPooling2D, Input, Dropout
from keras.models import Model, Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
df = pd.read_json("/kaggle/input/vehicle-number-plate-detection/Indian_Number_plates.json", lines=True)
df.head()
os.mkdir("Indian Number Plates")
dataset = dict()
dataset["image_name"] = list()
dataset["image_width"] = list()
dataset["image_height"] = list()
dataset["top_x"] = list()
dataset["top_y"] = list()
dataset["bottom_x"] = list()
dataset["bottom_y"] = list()

counter = 0
for index, row in df.iterrows():
    img = urllib.request.urlopen(row["content"])
    img = Image.open(img)
    img = img.convert('RGB')
    img.save("Indian Number Plates/licensed_car{}.jpeg".format(counter), "JPEG")
    
    dataset["image_name"].append("licensed_car{}".format(counter))
    
    data = row["annotation"]
    
    dataset["image_width"].append(data[0]["imageWidth"])
    dataset["image_height"].append(data[0]["imageHeight"])
    dataset["top_x"].append(data[0]["points"][0]["x"])
    dataset["top_y"].append(data[0]["points"][0]["y"])
    dataset["bottom_x"].append(data[0]["points"][1]["x"])
    dataset["bottom_y"].append(data[0]["points"][1]["y"])
    
    counter += 1
print("Downloaded {} car images.".format(counter))
dataset
df = pd.DataFrame(dataset)
df.head()
df.to_csv("indian_license_plates.csv", index=False)
df = pd.read_csv("indian_license_plates.csv")
df["image_name"] = df["image_name"] + ".jpeg"
df.drop(["image_width", "image_height"], axis=1, inplace=True)
df.head()
lucky_test_samples = np.random.randint(0, len(df), 5)
reduced_df = df.drop(lucky_test_samples, axis=0)
len(lucky_test_samples)
len(reduced_df)
WIDTH = 224
HEIGHT = 224
CHANNEL = 3

def show_img(index):
    image = cv2.imread("Indian Number Plates/" + df["image_name"].iloc[index])
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, dsize=(WIDTH, HEIGHT))

    tx = int(df["top_x"].iloc[index] * WIDTH)
    ty = int(df["top_y"].iloc[index] * HEIGHT)
    bx = int(df["bottom_x"].iloc[index] * WIDTH)
    by = int(df["bottom_y"].iloc[index] * HEIGHT)

    image = cv2.rectangle(image, (tx, ty), (bx, by), (0, 0, 255), 1)
    plt.imshow(image)
    plt.show()
show_img(5)
reduced_df.head()
lucky_test_samples
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K

# Create the model
model = Sequential()
model.add(VGG16(weights="imagenet", include_top=False, input_shape=(HEIGHT, WIDTH, CHANNEL)))
model.add(Flatten())
model.add(Dense(128, activation="relu"))
model.add(Dense(64, activation="relu"))
model.add(Dense(64, activation="relu"))
model.add(Dense(4, activation="sigmoid"))

model.layers[-6].trainable = False

model.summary()
data = []
for idx, row in reduced_df.iterrows():
    img = cv2.resize(cv2.imread("Indian Number Plates/" + row['image_name']) / 255.0, dsize=(WIDTH, HEIGHT))
    data.append(img)
data = np.array(data)
data.shape
X_train, X_validate = data[:210], data[210:]
X_train.shape, X_validate.shape
adam = Adam(lr=0.0005)
model.compile(optimizer=adam, loss="mse")
y_data = []
for idx, row in reduced_df.iterrows():
    y_data.append([row['top_x'], row['top_y'], row['bottom_x'], row['bottom_y']])
y_data = np.array(y_data)
y_data
Y_train,Y_Validate = y_data[:210], y_data[210:]
Y_train.shape, Y_Validate.shape
history = model.fit(X_train, Y_train, validation_data=(X_validate, Y_Validate), epochs=30, batch_size=21)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()
X_test = []
Y_test = []
for idx, row in df.iloc[lucky_test_samples].iterrows():
    img = cv2.resize(cv2.imread("Indian Number Plates/" + row['image_name']) / 255.0, dsize=(WIDTH, HEIGHT))
    X_test.append(img)
    Y_test.append([row['top_x'], row['top_y'], row['bottom_x'], row['bottom_y']])
    show_img(idx)
X_test = np.array(X_test)
Y_test = np.array(Y_test)
X_test.shape, Y_test.shape
model.evaluate(X_validate, Y_Validate, batch_size=1)
model.evaluate(X_test, Y_test, batch_size=1)
Y_hat = model.predict(X_test)
Y_hat.shape
def show_test_image(img, y_hat):
    tx = int(y_hat[0] * WIDTH)
    ty = int(y_hat[1] * HEIGHT)
    bx = int(y_hat[2] * WIDTH)
    by = int(y_hat[3] * HEIGHT)
    img = cv2.cvtColor(img.astype(np.float32), cv2.COLOR_BGR2RGB)
    image = cv2.rectangle(img, (tx, ty), (bx, by), (0, 0, 255), 1)
    plt.imshow(image)
    plt.show()
for i in range(0, len(Y_hat)):
    y_hat = Y_hat[i]
    img = X_test[i]
    show_test_image(img, y_hat)