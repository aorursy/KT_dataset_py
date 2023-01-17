import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Activation, Dense, Dropout, Flatten
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
import matplotlib.pyplot as plt
from glob import glob

img = load_img("../input/104-flowers-garden-of-eden/jpeg-192x192/train/azalea/10380.jpeg")

plt.imshow(img)
plt.show()
x = img_to_array(img)
print(x.shape)
className = glob("../input/104-flowers-garden-of-eden/jpeg-192x192/train" + '/*')
number_of_class = len(className)
number_of_class
model = Sequential()

model.add(Conv2D(32,(3,3), input_shape = x.shape))
model.add(Activation("relu"))
model.add(MaxPooling2D())

model.add(Conv2D(32,(3,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D())

model.add(Conv2D(64,(3,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D())

model.add(Conv2D(128,(3,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D())

model.add(Conv2D(256,(3,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D())

model.add(Flatten())
model.add(Dense(1024))
model.add(Activation("relu"))
model.add(Dropout(0.5))
model.add(Dense(number_of_class))
model.add(Activation("softmax"))
model.compile(loss = "categorical_crossentropy",
             optimizer = "Adam",
             metrics = ["accuracy"])
a = glob("../input/104-flowers-garden-of-eden/jpeg-192x192/train" + "/alpine sea holly" +'/*')

print("Yaklaşık bu kadar çicek var =" ,len(a) * number_of_class)
print("Bu yeterli değil. Deep learning için data generation yapıcaz.")
b = glob("../input/104-flowers-garden-of-eden/jpeg-192x192/val" + "/alpine sea holly" +'/*')

print("Yaklaşık bu kadar çicek var =" ,len(b) * number_of_class)
print("Bu yeterli değil. Deep learning için data generation yapıcaz.")
train_datagen = ImageDataGenerator(rescale=1./255,
                  shear_range=0.3,
                  horizontal_flip=True,
                  zoom_range=0.3)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    "../input/104-flowers-garden-of-eden/jpeg-192x192/train", 
    target_size=(192, 192),
    batch_size = 32,
    color_mode="rgb",
    class_mode = "categorical")

test_generator = test_datagen.flow_from_directory(
    "../input/104-flowers-garden-of-eden/jpeg-192x192/val", 
    target_size=(192, 192),
    batch_size = 32,
    color_mode="rgb",
    class_mode = "categorical")
step_size_train=train_generator.n//train_generator.batch_size
step_size_test=test_generator.n//test_generator.batch_size
hist = model.fit_generator(
    generator = train_generator,
    steps_per_epoch = step_size_train,
    epochs=5,
    validation_data=test_generator,
    validation_steps = step_size_test)
print(hist.history.keys())

plt.plot(hist.history["loss"], label="Train Loss")
plt.plot(hist.history["val_loss"], label="Validation loss")
plt.legend()
plt.show()

plt.plot(hist.history["accuracy"], label="Train accuracy")
plt.plot(hist.history["val_accuracy"], label="Validation accuracy")
plt.legend()
plt.show()
from PIL import Image,ImageFilter,ImageEnhance
from pylab import *
img = load_img("../input/104-flowers-garden-of-eden/jpeg-192x192/train/anthurium/10104.jpeg")
plt.imshow(img)
plt.show()
img = Image.open('../input/104-flowers-garden-of-eden/jpeg-192x192/train/anthurium/10104.jpeg').convert('L') # gri yapar.
img
img = array(Image.open('../input/104-flowers-garden-of-eden/jpeg-192x192/train/anthurium/10104.jpeg').convert('L'))

figure()
gray()
contour(img, origin='image')
axis('equal')
axis('off')
image = Image.open(r"../input/104-flowers-garden-of-eden/jpeg-192x192/train/anthurium/10104.jpeg").convert("L")

image = image.filter(ImageFilter.FIND_EDGES)

image
# sahip olunan resimlere gürültü ekleyerek yeni resimler oluşturduk.
# We created new pictures by adding noise to the owned pictures.

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img 
from matplotlib import pyplot

datagen = ImageDataGenerator(rotation_range=40, width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2, zoom_range=0.2, horizontal_flip=True, vertical_flip=True, fill_mode='nearest')
img = load_img("../input/104-flowers-garden-of-eden/jpeg-192x192/train/anthurium/10104.jpeg") 
x = img_to_array(img)
x = x.reshape((1,) + x.shape) 
it = datagen.flow(x, batch_size=1)

for i in range(1,5):
    batch = it.next()
    image = batch[0].astype('uint8')
    pyplot.imshow(image)
    pyplot.show()
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from keras.models import Sequential
from keras.layers import Dense
from keras.applications.vgg16 import VGG16
import matplotlib.pyplot as plt
from glob import glob
vgg = VGG16()

print(vgg.summary())
print(type(vgg))
# predictions (Dense)          (None, 1000)              4097000   

# 1000 Class used. I don't have that much class, I'll adjust it for myself.

# 1000 Sınıf kullanılmış. bende o kadar class yok kendime göre ayarlıyacağım.

vgg_layer_list = vgg.layers

model = Sequential()
for i in range(len(vgg_layer_list)-1):
    model.add(vgg_layer_list[i])

print(model.summary())

# predictions (Dense)          (None, 1000)              4097000   
# We took off. 
# Çıkardık.
for layers in model.layers:
    layers.trainable = False

model.add(Dense(number_of_class, activation="softmax"))

# dense_3 (Dense)              (None, 104)               17292 
# we added
# ekledik
print(model.summary())
model.compile(loss = "categorical_crossentropy",
              optimizer = "rmsprop",
              metrics = ["accuracy"])
train_data = ImageDataGenerator().flow_from_directory("../input/104-flowers-garden-of-eden/jpeg-224x224/train",target_size = (224,224))
test_data = ImageDataGenerator().flow_from_directory("../input/104-flowers-garden-of-eden/jpeg-224x224/val",target_size = (224,224))
hist = model.fit_generator(train_data,
                           steps_per_epoch=1600//32,
                           epochs= 3,
                           validation_data=test_data,
                           validation_steps= 800//32)
print(hist.history.keys())
plt.plot(hist.history["loss"],label = "training loss")
plt.plot(hist.history["val_loss"],label = "validation loss")
plt.legend()
plt.show()
plt.figure()
plt.plot(hist.history["accuracy"],label = "training acc")
plt.plot(hist.history["val_accuracy"],label = "validation acc")
plt.legend()
plt.show()
trdata = ImageDataGenerator()
traindata = trdata.flow_from_directory(directory="../input/104-flowers-garden-of-eden/jpeg-224x224/train",target_size=(224,224))
tsdata = ImageDataGenerator()
testdata = tsdata.flow_from_directory(directory="../input/104-flowers-garden-of-eden/jpeg-224x224/val", target_size=(224,224))
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten
from keras.layers import Conv2D
from keras.layers import MaxPool2D
model = Sequential()
model.add(Conv2D(input_shape=(224,224,3),filters=64,kernel_size=(3,3),padding="same", activation="relu"))
model.add(Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

model.add(Conv2D(128, (3,3), padding="same", activation="relu"))
model.add(Conv2D(128, (3,3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

model.add(Conv2D(256,(3,3), padding="same", activation="relu"))
model.add(Conv2D(256, (3,3), padding="same", activation="relu"))
model.add(Conv2D(256, (3,3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

model.add(Conv2D(512, (3,3), padding="same", activation="relu"))
model.add(Conv2D(512, (3,3), padding="same", activation="relu"))
model.add(Conv2D(512, (3,3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

model.add(Conv2D(512, (3,3), padding="same", activation="relu"))
model.add(Conv2D(512, (3,3), padding="same", activation="relu"))
model.add(Conv2D(512, (3,3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
model.add(Flatten())
model.add(Dense(units=4096,activation="relu"))
model.add(Dense(units=4096,activation="relu"))
model.add(Dense(units=number_of_class, activation="softmax"))
model.compile(loss=keras.losses.categorical_crossentropy, optimizer="adam", metrics=["accuracy"])
model.summary()
hist = model.fit_generator(train_data,
                           steps_per_epoch=1600//32,
                           epochs= 2,
                           validation_data=test_data,
                           validation_steps= 800//32)
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
from keras.applications.vgg16 import VGG16 
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
from keras.applications.vgg16 import VGG16
vgg = VGG16()
model = VGG16()

image = load_img("../input/104-flowers-garden-of-eden/jpeg-224x224/train/azalea/10326.jpeg", target_size=(224, 224))

image = img_to_array(image)

image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))

image = preprocess_input(image)

yhat = model.predict(image)

label = decode_predictions(yhat)

label = label[0][0]

print(label[1], label[2]*100)
model = VGG16()

image = load_img("../input/104-flowers-garden-of-eden/jpeg-224x224/train/azalea/10326.jpeg", target_size=(224, 224))
plt.imshow(img)
plt.show()

image = img_to_array(image)

image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))

image = preprocess_input(image)

yhat = model.predict(image)

label = decode_predictions(yhat)

label = label[0][0]

print(label[1], label[2]*100)
model = VGG16()

image = load_img("../input/104-flowers-garden-of-eden/jpeg-224x224/train/corn poppy/11135.jpeg", target_size=(224, 224))
plt.imshow(image)
plt.show()

image = img_to_array(image)

image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))

image = preprocess_input(image)

yhat = model.predict(image)

label = decode_predictions(yhat)

label = label[0][0]

print(label[1], label[2]*100)
model = VGG16()

image = load_img("../input/104-flowers-garden-of-eden/jpeg-224x224/train/black-eyed susan/10405.jpeg", target_size=(224, 224))
plt.imshow(image)
plt.show()

image = img_to_array(image)

image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))

image = preprocess_input(image)

yhat = model.predict(image)

label = decode_predictions(yhat)

label = label[0][0]

print(label[1], label[2]*100)