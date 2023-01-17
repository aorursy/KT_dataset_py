import numpy as np

import cv2

from matplotlib import pyplot as plt

from glob import glob

import os

from bs4 import BeautifulSoup as bs

from IPython.display import HTML, display

import time



from sklearn.model_selection import train_test_split

from sklearn.preprocessing import OneHotEncoder



import keras as keras

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
dataset = []

classList = []

for folder in sorted(glob("../input/images/Images/*")):

    path = folder[23:]

    className = folder[33:]

    classList.append(className)

    for imgPath in glob(folder + "/*"):

        bareName = imgPath[24+len(path):2+imgPath[2:].index('.')]

        dataset.append((path, bareName, className))
i = 0

bar = display(progress(i, len(dataset)), display_id=True)

x = []

y = []

for (path, bareName, nama) in dataset:

    img = cv2.imread("../input/images/Images/" + path + "/" + bareName + ".jpg")

    anotasiArr = load_anotasi("../input/annotations/Annotation/" + path + "/" + bareName)

    for anotasi in anotasiArr:

        x1, x2, y1, y2 = anotasi

        img2 = preprocessing(img[x1:x2, y1:y2])

        x.append(img2)

        y.append(nama)

    i += 1

    bar.update(progress(i, len(dataset)))

  

x = np.array(x)

y = np.array(y)
onehot_encoder = OneHotEncoder(sparse=False)

y2 = np.expand_dims(y, axis=1)

y2 = onehot_encoder.fit_transform(y2)

x_train, x_test, y_train, y_test = train_test_split(x, y2, test_size=0.3)
from keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions



mobilenet = MobileNetV2(include_top=True, weights=None, input_shape=np.shape(x[0]), classes=120)

mobilenet.compile(loss=keras.losses.categorical_crossentropy, optimizer='adam', metrics=['accuracy'])

#mobilenet.summary()
augs_gen = ImageDataGenerator(

        #rescale=1./255,

        featurewise_center=False,

        samplewise_center=False, 

        featurewise_std_normalization=False,  

        samplewise_std_normalization=False,  

        zca_whitening=False,  

        rotation_range=10,  

        zoom_range = 0.1, 

        width_shift_range=0.2,  

        height_shift_range=0.2, 

        horizontal_flip=True,  

        vertical_flip=False) 
waktu = time.time()

h = mobilenet.fit_generator(

    augs_gen.flow(x_train, y_train),

    steps_per_epoch = 500,

    epochs = 200,

    validation_data=(x_test, y_test)

)

print(time.time() - waktu)
h.history
import urllib



#chihuahua

#url = "https://s3.amazonaws.com/cdn-origin-etr.akc.org/wp-content/uploads/2017/11/12234710/Chihuahua-On-White-03.jpg"

#poddle

url = "https://i.pinimg.com/originals/4f/3d/70/4f3d7030404d5982105f96b8118292c8.jpg"

#pug

#url = "https://vetstreet.brightspotcdn.com/dims4/default/354d0cf/2147483647/thumbnail/645x380/quality/90/?url=https%3A%2F%2Fvetstreet-brightspot.s3.amazonaws.com%2Fdc%2Fc4%2F8ccd3a28438d81b2f2f5d8031a05%2Fpug-ap-r82p3q-645.jpg"



resp = urllib.request.urlopen(url)

img = np.asarray(bytearray(resp.read()), dtype="uint8")

img = cv2.imdecode(img, cv2.IMREAD_COLOR)

img = preprocessing(img)

tampil(img)



print(np.shape(img))

print(np.shape(x[0]))



predict = np.expand_dims(img, axis=0)

predict = mobilenet.predict(predict)



predict = onehot_encoder.inverse_transform(predict)

print(predict)
