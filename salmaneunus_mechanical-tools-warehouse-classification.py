!pip install pillow
import PIL

print('Pillow Version:', PIL.__version__)
from PIL import Image

img1= Image.open("../input/mechanical-tools-dataset/Mechanical Tools Image dataset/Mechanical Tools Image dataset/pliers/000000.jpg")
print(img1.format,img1.size,img1.mode)
# load and display an image with Matplotlib

from matplotlib import image

from matplotlib import pyplot

# load image as pixel array

data = image.imread('../input/mechanical-tools-dataset/Mechanical Tools Image dataset/Mechanical Tools Image dataset/pliers/000000.jpg')

# summarize shape of the pixel array

print(data.dtype)

print(data.shape)

# display the array of pixels as an image

pyplot.imshow(data)

pyplot.show()
from PIL import Image

from numpy import asarray

# load the image

image = Image.open('../input/mechanical-tools-dataset/Mechanical Tools Image dataset/Mechanical Tools Image dataset/Gasoline Can/Gasoline can (101).jpg')

# convert image to numpy array

data = asarray(image)

# summarize shape

print(data.shape)

# create Pillow image

image2 = Image.fromarray(data)

# summarize image details

print(image2.format)

print(image2.mode)

print(image2.size)
pyplot.imshow(image2)
# load and display an image with Matplotlib

from matplotlib import image

from matplotlib import pyplot

# load image as pixel array

data = image.imread('../input/mechanical-tools-dataset/Mechanical Tools Image dataset/Mechanical Tools Image dataset/Hammer/000149.jpg')

# summarize shape of the pixel array

print(data.dtype)

print(data.shape)

# display the array of pixels as an image

pyplot.imshow(data)

pyplot.show()
# load and display an image with Matplotlib

from matplotlib import image

from matplotlib import pyplot

# load image as pixel array

data = image.imread('../input/mechanical-tools-dataset/Mechanical Tools Image dataset/Mechanical Tools Image dataset/Wrench/000010.jpg')

# summarize shape of the pixel array

print(data.dtype)

print(data.shape)

# display the array of pixels as an image

pyplot.imshow(data)

pyplot.show()
# create a thumbnail of an image

from PIL import Image

# load the image

image = Image.open('../input/mechanical-tools-dataset/Mechanical Tools Image dataset/Mechanical Tools Image dataset/Hammer/000013.jpg')

# shows the size of the image

print(image.size)

# create a thumbnail and preserve aspect ratio

image.thumbnail((100,100))

# report the size of the thumbnail

print(image.size)
image = Image.open('../input/mechanical-tools-dataset/Mechanical Tools Image dataset/Mechanical Tools Image dataset/Gasoline Can/Gasoline can (105).jpg')

# report the size of the image

print(image.size)

# resize image and ignore original aspect ratio

img_resized = image.resize((200,200))

# report the size of the thumbnail

print(img_resized.size)
# load image

image = Image.open('../input/mechanical-tools-dataset/Mechanical Tools Image dataset/Mechanical Tools Image dataset/Hammer/000110.jpg')

# horizontal flip

hoz_flip = image.transpose(Image.FLIP_LEFT_RIGHT)

# vertical flip

ver_flip = image.transpose(Image.FLIP_TOP_BOTTOM)

# plot all three images using matplotlib

pyplot.subplot(311)

pyplot.imshow(image)

pyplot.subplot(312)

pyplot.imshow(hoz_flip)

pyplot.subplot(313)

pyplot.imshow(ver_flip)

pyplot.show()
# load image

image = Image.open('../input/mechanical-tools-dataset/Mechanical Tools Image dataset/Mechanical Tools Image dataset/Toolbox/000023.jpg')

# plot original image

pyplot.subplot(311)

pyplot.imshow(image)

# rotate 45 degrees

pyplot.subplot(312)

pyplot.imshow(image.rotate(45))

# rotate 90 degrees

pyplot.subplot(313)

pyplot.imshow(image.rotate(90))

pyplot.show()

#rotates 270 degrees

pyplot.imshow(image.rotate(270))

pyplot.show()
from PIL import Image

# load image

image = Image.open('../input/mechanical-tools-dataset/Mechanical Tools Image dataset/Mechanical Tools Image dataset/Hammer/000014.jpg')

# create a cropped image

cropped = image.crop((100, 100, 200, 200))

# show cropped image

pyplot.imshow(cropped)
import keras

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten

from keras.layers import Conv2D, MaxPooling2D

from keras.utils import to_categorical

from keras.preprocessing import image

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from tqdm import tqdm

%matplotlib inline
train = pd.read_csv('../input/mechanical-tools-dataset/Annotated.csv')    # reading the csv file

train.head()  
train.head(20)
train.tail()
train.shape
train.columns
train_image1 = []

for i in range(0,17):

    img1 = image.load_img('../input/mechanical-tools-dataset/Mechanical Tools Image dataset/Mechanical Tools Image dataset/Screw Driver/'+train['Id'][i],target_size=(400,400,3))

    img1 = image.img_to_array(img1)

    img1 = img1/255

    train_image1.append(img1)

train_image2 = []

for i in range(17,33):

    img2 = image.load_img('../input/mechanical-tools-dataset/Mechanical Tools Image dataset/Mechanical Tools Image dataset/Hammer/'+train['Id'][i],target_size=(400,400,3))

    img2 = image.img_to_array(img2)

    img2 = img2/255

    train_image2.append(img2)

train_image3 = []

for i in range(34,50):

    img3 = image.load_img('../input/mechanical-tools-dataset/Mechanical Tools Image dataset/Mechanical Tools Image dataset/pliers/'+train['Id'][i],target_size=(400,400,3))

    img3 = image.img_to_array(img3)

    img3 = img3/255

    train_image3.append(img3)

train_image4 = []

for i in range(46,75):

    img4 = image.load_img('../input/mechanical-tools-dataset/Mechanical Tools Image dataset/Mechanical Tools Image dataset/Rope/'+train['Id'][i],target_size=(400,400,3))

    img4 = image.img_to_array(img4)

    img4 = img4/255

    train_image4.append(img4)

train_image5 = []

for i in range(71,91):

    img5 = image.load_img('../input/mechanical-tools-dataset/Mechanical Tools Image dataset/Mechanical Tools Image dataset/Toolbox/'+train['Id'][i],target_size=(400,400,3))

    img5 = image.img_to_array(img5)

    img5 = img5/255

    train_image5.append(img5)

X = np.array(train_image1+train_image2+train_image3+train_image4+train_image5)
X.shape
plt.imshow(X[8])
plt.imshow(X[17])
plt.imshow(X[35])
plt.imshow(X[50])
plt.imshow(X[60])
plt.imshow(X[70])
plt.imshow(X[2])
plt.imshow(X[6])
plt.imshow(X[69])
plt.imshow(X[86])
plt.imshow(X[90])
y = np.array(train.drop(['Label','Id'],axis=1))
y.shape
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.1)
model = Sequential()

model.add(Conv2D(filters=16, kernel_size=(5, 5), activation="relu", input_shape=(400,400,3)))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(filters=32, kernel_size=(5, 5), activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(filters=64, kernel_size=(5, 5), activation='relu'))

model.add(Flatten())

model.add(Dense(128, activation='relu'))

model.add(Dense(6, activation='sigmoid'))
model.summary()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=5, validation_data=(X_test, y_test), batch_size=16)
img = image.load_img('../input/mechanical-tools-dataset/Mechanical Tools Image dataset/Mechanical Tools Image dataset/pliers/000310.jpg',target_size=(400,400,3))

img = image.img_to_array(img)

img = img/255
plt.imshow(img)
plt.imshow(X_test[2])
scores = model.evaluate(X_test, y_test, verbose=0)

print("Accuracy: %.2f%%" % (scores[1]*100))