import numpy as np

import cv2

from matplotlib import pyplot as plt

from glob import glob

import os

from bs4 import BeautifulSoup as bs

from IPython.display import HTML, display

from time import time



from sklearn.model_selection import train_test_split

from sklearn.preprocessing import OneHotEncoder



import keras

from keras.models import Sequential

from keras.layers import Dense, Activation, Dropout, Flatten

from keras.layers import Conv2D, GlobalAveragePooling2D

from keras.layers import MaxPooling2D, BatchNormalization

from keras.preprocessing.image import ImageDataGenerator



def tampil(img):

  plt.figure(figsize=(10,10))

  plt.axis("off")

  plt.imshow(img,cmap="gray")

  

def load_anotasi(path):

  handler = open(path).read()

  soup = bs(handler,"xml")

  cords = []

  for message in soup.findAll('bndbox'):

    cords.append([int(message.ymin.text),int(message.ymax.text),int(message.xmin.text),int(message.xmax.text)])

  return cords



def preprocessing(img):

    img = cv2.resize(img, (128, 128))

    #img = cv2.blur(img, (3, 3))

    #https://stackoverflow.com/questions/31998428/opencv-python-equalizehist-colored-image

    img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)

    img[:,:,0] = cv2.equalizeHist(img[:,:,0])

    img = cv2.cvtColor(img, cv2.COLOR_YUV2BGR)

    return img



def progress(value, max=100):

    return HTML("""

        <progress

            value='{value}'

            max='{max}',

            style='width: 100%'

        >

            {value}

        </progress>

    """.format(value=value, max=max))

def preprocessing(img):

    img = cv2.resize(img, (128, 128))

    #img = cv2.blur(img, (3, 3))

    #https://stackoverflow.com/questions/31998428/opencv-python-equalizehist-colored-image

    img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)

    img[:,:,0] = cv2.equalizeHist(img[:,:,0])

    img = cv2.cvtColor(img, cv2.COLOR_YUV2BGR)

    return img



def progress(value, max=100):

    return HTML("""

        <progress

            value='{value}'

            max='{max}',

            style='width: 100%'

        >

            {value}

        </progress>

    """.format(value=value, max=max))
dataset = []

classList = []

for folder in sorted(glob("../input/images/Images/*")):

    path = folder[23:]

    className = folder[33:]

    classList.append(className)

    for imgPath in glob(folder + "/*"):

        bareName = imgPath[24+len(path):2+imgPath[2:].index('.')]

        dataset.append((path, bareName, className))
from keras.applications.xception import Xception, preprocess_input, decode_predictions



pretrained = Xception(include_top = False, pooling='avg')

pretrained.summary()
from keras.preprocessing.image import load_img, img_to_array

from time import time



waktu = time()

i = 0

benar = 0

bar = display(progress(i, len(dataset)), display_id=True)

x = []

y = []

for data in dataset:

  (path, bareName, nama) = data

  imgPath = "../input/images/Images/" + path + "/" + bareName + ".jpg"

  img = cv2.imread(imgPath)

  anotasiArr = load_anotasi("../input/annotations/Annotation/" + path + "/" + bareName)

  for anotasi in anotasiArr:

    x1, x2, y1, y2 = anotasi

    img2 = img[x1:x2, y1:y2]

    img2 = cv2.resize(img2, (299, 299))

    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

    img2 = np.expand_dims(img2, axis=0)

    img2 = preprocess_input(img2)



    predict = pretrained.predict(img2)

    #predict = decode_predictions(predict, top=1)[0][0][1]

    #if (predict == nama):

    #  benar = benar + 1

    x.append(predict)

    #x.append(img2)

    y.append(nama)

    #break

  i += 1

  bar.update(progress(i, len(dataset)))

  #break

print(time() - waktu)

x = np.array(x)

y = np.array(y)
model = Sequential()

model.add(Dense(1024, activation='relu', input_shape=np.shape(x[0])))

model.add(Flatten())

model.add(Dense(512, activation='relu'))

model.add(Dense(256, activation='relu'))

model.add(Dense(120,activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy, optimizer='adam', metrics=['accuracy'])

model.summary()
onehot_encoder = OneHotEncoder(sparse=False)

y2 = np.expand_dims(y, axis=1)

y2 = onehot_encoder.fit_transform(y2)



x_train, x_test, y_train, y_test = train_test_split(x, y2, test_size = 0.2, random_state=0)
waktu = time()

h = model.fit(x_train, y_train, epochs = 100, validation_data=(x_test, y_test))

print(time() - waktu)
import matplotlib.pyplot as plt



plt.plot(h.history["acc"])

plt.plot(h.history["val_acc"])

plt.title('Akurasi')

plt.xlabel('epoch')

plt.legend(['Train Acc', 'Val Acc'])
import urllib



#chihuahua

#url = "https://s3.amazonaws.com/cdn-origin-etr.akc.org/wp-content/uploads/2017/11/12234710/Chihuahua-On-White-03.jpg"

#poddle

#url = "https://i.pinimg.com/originals/4f/3d/70/4f3d7030404d5982105f96b8118292c8.jpg"

#pug

url = "https://vetstreet.brightspotcdn.com/dims4/default/354d0cf/2147483647/thumbnail/645x380/quality/90/?url=https%3A%2F%2Fvetstreet-brightspot.s3.amazonaws.com%2Fdc%2Fc4%2F8ccd3a28438d81b2f2f5d8031a05%2Fpug-ap-r82p3q-645.jpg"



resp = urllib.request.urlopen(url)

img = np.asarray(bytearray(resp.read()), dtype="uint8")

img = cv2.imdecode(img, cv2.IMREAD_COLOR)

tampil(img)

img = cv2.resize(img, (299, 299))

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

img = np.expand_dims(img, axis=0)

img = preprocess_input(img)

predict = pretrained.predict(img)



print(np.shape(predict))

print(np.shape(x[0]))



predict = np.expand_dims(predict, axis=0)

predict = model.predict(predict)



predict = onehot_encoder.inverse_transform(predict)

print(predict)
