import numpy as np

import matplotlib.pyplot as plt

import keras

from keras.utils.np_utils import to_categorical

from keras.models import Sequential

from keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D

from keras.optimizers import Adam

import pickle as pkl
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
np.random.seed(0)
with open('/kaggle/input/jadslimgermantrafficsigns/jadslim-german-traffic-signs-a11dc223e390/train.p', 'rb') as f:

    train_data =pkl.load(f)



with open('/kaggle/input/jadslimgermantrafficsigns/jadslim-german-traffic-signs-a11dc223e390/test.p', 'rb') as f:

  test_data =pkl.load(f)

  

with open('/kaggle/input/jadslimgermantrafficsigns/jadslim-german-traffic-signs-a11dc223e390/valid.p', 'rb') as f:

  val_data =pkl.load(f)

  

X_train, y_train = train_data['features'],train_data['labels']

X_test, y_test = test_data['features'],test_data['labels']

X_val, y_val = val_data['features'],val_data['labels']
train_data.keys()
import pandas as pd

data = pd.read_csv('/kaggle/input/jadslimgermantrafficsigns/jadslim-german-traffic-signs-a11dc223e390/signnames.csv')
data.tail()
data.info()
len(y_train)
num_of_samples =[]

cols = 5

num_classes = 43

fig, axs = plt.subplots(nrows= num_classes, ncols=cols, figsize= (15,50))

fig.tight_layout()



for i in range(cols):

  for j,raw in data.iterrows():

    X_selected = X_train[y_train== j]

    axs[j][i].imshow(X_selected[np.random.randint(0,(len(X_selected)-1)), :, :], cmap=plt.get_cmap('gray'))

    axs[j][i].axis('off')



    if i == 2:

      axs[j][i].set_title(str(j)+ '-'+raw["SignName"])

      num_of_samples.append(len(X_selected))

print(num_of_samples)

plt.figure(figsize=(12,4))

plt.bar(range(num_classes), num_of_samples)

plt.title("Distribution of the train dataset")

plt.xlabel("Class number")

plt.ylabel("Number of images")

plt.show()
import cv2

plt.imshow(X_train[1000])

plt.axis('off')

print(X_train[1000].shape)

print(y_train[1000])
def grayscale(img):

  img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

  return img
img = grayscale(X_train[1000])

plt.imshow(img, cmap="gray")

plt.axis('off')

print(img.shape)
def equalize(img):

  img = cv2.equalizeHist(img)

  return img
img - equalize(img)

plt.imshow(img, cmap="gray")

plt.axis('off')

print(img.shape)
def preprocess(img):

  img=  grayscale(img)

  img=  equalize(img)

  img = img/255

  return img
X_train = np.array(list(map(preprocess,X_train)))

X_test = np.array(list(map(preprocess,X_test)))

X_val = np.array(list(map(preprocess,X_val)))
len(num_of_samples)
plt.imshow(X_train[np.random.randint(0,len(X_train)-1)])

plt.axis('off')

print(X_train.shape)
X_train = X_train.reshape(34799,32,32,1)

X_test = X_test.reshape(12630,32,32,1)

X_val = X_val.reshape(4410,32,32,1)
from keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(width_shift_range=0.1,

                            height_shift_range=0.1,

                            zoom_range= 0.2,

                            shear_range=0.1,

                            rotation_range=10)

datagen.fit(X_train)

y_train = to_categorical(y_train,43)

y_test = to_categorical(y_test,43)

y_val = to_categorical(y_val,43)
data.iterrows
X_train.shape
def le_net():

  model = Sequential()

  model.add(Conv2D(30,(5,5), input_shape=(32,32,1), activation = 'relu'))

  model.add(MaxPooling2D(pool_size=(2,2)))

  model.add(Conv2D(15,(3,3), activation='relu'))

  model.add(MaxPooling2D(pool_size=(2,2)))

  model.add(Flatten())

  model.add(Dense(500,activation="relu"))

  model.add(Dropout(0.5))

  model.add(Dropout(0.5))

  model.add(Dense(num_classes,activation="softmax"))

  model.compile(Adam(lr=0.001),loss ='categorical_crossentropy', metrics=['accuracy'])

  return model
lenet = le_net()

lenet.summary()
history= lenet.fit(X_train,y_train, validation_data=(X_val,y_val), epochs=20, validation_split= 0.1, batch_size=400,verbose=1,shuffle=1)
lenet.save('digit.h5')
!ls
# from google.colab import files
# files.download('digit.h5')
plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.legend(['loss','val_loss'])

plt.title('Loss')

plt.xlabel('epoch')
plt.plot(history.history['accuracy'])

plt.plot(history.history['val_accuracy'])

plt.legend(['accuracy','val_accuracy'])

plt.title('Accuracy')

plt.xlabel('epoch')

def modified_model():

  model = Sequential()

  model.add(Conv2D(60,(5,5), input_shape=(32,32,1), activation = 'relu'))

  model.add(Conv2D(60,(5,5), activation='relu'))

  model.add(MaxPooling2D(pool_size=(2,2)))

  

  

  model.add(Conv2D(30,(3,3), activation='relu'))

  model.add(Conv2D(30,(3,3), activation='relu'))

  model.add(MaxPooling2D(pool_size=(2,2)))

  model.add(Dropout(0.5))

  

  model.add(Flatten())

  model.add(Dense(500,activation="relu"))

  model.add(Dropout(0.5))

  model.add(Dropout(0.5))

  model.add(Dense(num_classes,activation="softmax"))

  model.compile(Adam(lr=0.001),loss ='categorical_crossentropy', metrics=['accuracy'])

  return model
lenet2 = modified_model()
lenet2.summary()
history= lenet2.fit(X_train,y_train, validation_data=(X_val,y_val), epochs=20, validation_split= 0.1, batch_size=400,verbose=1,shuffle=1)
lenet2.save('traffic_sign_model.h5')
!ls
# files.download('traffic_sign_model.h5')
plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.legend(['loss','val_loss'])

plt.title('Loss')

plt.xlabel('epoch')
plt.plot(history.history['accuracy'])

plt.plot(history.history['val_accuracy'])

plt.legend(['acc','val_accuracy'])

plt.title('Accuracy')

plt.xlabel('epoch')

import requests

from PIL import Image

url = "https://2qrvzf20gyjyyrbe93pemsx1-wpengine.netdna-ssl.com/wp-content/uploads/2016/07/Stop-Sign-400.jpg"

response = requests.get(url,stream=True)

img = Image.open(response.raw).convert("L")

plt.imshow(img,cmap="gray")
import cv2

img_array = np.asarray(img)

res = cv2.resize(img_array,(32,32))

plt.imshow(res,cmap="gray")
res = res/255

res = res.reshape(1,32,32,1)

lenet2.predict_classes(res)

# lenet.predict(res)