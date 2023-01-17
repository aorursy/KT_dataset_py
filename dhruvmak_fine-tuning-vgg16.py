import matplotlib

matplotlib.use("Agg")



import tensorflow as tf

import matplotlib.pyplot as plt

from keras.preprocessing.image import ImageDataGenerator

from keras.optimizers import Adam

from keras.layers import Input

from keras.models import Model

from keras.applications import VGG16

from keras.layers.core import Flatten, Dense, Dropout

import numpy as np

import os
print("[Info] loading imagenet weights...")

baseModel = VGG16(weights="imagenet", include_top=False,

	input_tensor=Input(shape=(224, 224, 3)))



headModel = baseModel.output

headModel = Flatten(name="flatten")(headModel)

headModel = Dense(1024, activation='relu')(headModel)

headModel = Dropout(0.5)(headModel)

headModel = Dense(1, activation='sigmoid')(headModel)



model = Model(inputs=baseModel.input, outputs=headModel)

for layer in baseModel.layers:

	layer.trainable = False
#image preprocessing

train_datagen = ImageDataGenerator(rescale=1./255)

test_datagen = ImageDataGenerator(rescale=1./255)
batch_size = 32

print("[INFO] loading images...")

train_data_dir = "../input/real-life-industrial-dataset-of-casting-product/casting_data/train"     #directory of training data

test_data_dir = "../input/real-life-industrial-dataset-of-casting-product/casting_data/test"      #directory of test data
training_set = train_datagen.flow_from_directory(train_data_dir, 

                                                 target_size=(224, 224),

                                                 batch_size=batch_size, 

                                                 class_mode='binary')

test_set = test_datagen.flow_from_directory(test_data_dir, 

                                            target_size=(224, 224),

                                            batch_size=batch_size, 

                                            class_mode='binary')
print(training_set.class_indices)
print("[INFO] compiling model...")

opt = Adam(lr=0.001)

model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])
print("[INFO] training model...")

H = model.fit_generator(training_set,

                        steps_per_epoch=training_set.samples//batch_size,

                        validation_data=test_set,

                        epochs=5,

                        validation_steps=test_set.samples//batch_size)
print("[Info] serializing network...")

model.save("VGG16.hdf5")
%matplotlib inline

print("[Info] visualising model...")

plt.style.use("ggplot")

plt.figure()

plt.plot(np.arange(0, 5), H.history["loss"], label="train_loss")

plt.plot(np.arange(0, 5), H.history["val_loss"], label="val_loss")

plt.plot(np.arange(0, 5), H.history["accuracy"], label="train_accuracy")

plt.plot(np.arange(0, 5), H.history["val_accuracy"], label="val_accuracy")

plt.title("Training Loss and Accuracy")

plt.xlabel("Epoch #")

plt.ylabel("Loss/Accuracy")

plt.legend()

plt.show()

plt.savefig("VGG16.png")
import cv2

from keras.preprocessing import image

%matplotlib inline

imagepath = "../input/real-life-industrial-dataset-of-casting-product/casting_data/test/def_front/cast_def_0_1191.jpeg"

img = cv2.imread(imagepath)

img = cv2.resize(img, (224, 224))

orig = img.copy()

img = image.img_to_array(img)

img = np.expand_dims(img, axis=0)

img = img/255



print("[Info] predicting output")

prediction = model.predict(img)

if (prediction<0.5):

    print("def_front")

    cv2.putText(orig, "def_front", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

else:

    print("ok_front")

    cv2.putText(orig, "ok_front", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    

plt.imshow(orig)

plt.axis('off')

plt.show()
%matplotlib inline

imagepath = "../input/real-life-industrial-dataset-of-casting-product/casting_data/test/def_front/cast_def_0_1203.jpeg"

img = cv2.imread(imagepath)

img = cv2.resize(img, (224, 224))

orig = img.copy()

img = image.img_to_array(img)

img = np.expand_dims(img, axis=0)

img = img/255



print("[Info] predicting output")

prediction = model.predict(img)

if (prediction<0.5):

    print("def_front")

    cv2.putText(orig, "def_front", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

else:

    print("ok_front")

    cv2.putText(orig, "ok_front", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    

plt.imshow(orig)

plt.axis('off')

plt.show()
%matplotlib inline

imagepath = "../input/real-life-industrial-dataset-of-casting-product/casting_data/test/def_front/cast_def_0_1134.jpeg"

img = cv2.imread(imagepath)

img = cv2.resize(img, (224, 224))

orig = img.copy()

img = image.img_to_array(img)

img = np.expand_dims(img, axis=0)

img = img/255



print("[Info] predicting output")

prediction = model.predict(img)

if (prediction<0.5):

    print("def_front")

    cv2.putText(orig, "def_front", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

else:

    print("ok_front")

    cv2.putText(orig, "ok_front", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    

plt.imshow(orig)

plt.axis('off')

plt.show()
%matplotlib inline

imagepath = "../input/real-life-industrial-dataset-of-casting-product/casting_data/test/ok_front/cast_ok_0_1092.jpeg"

img = cv2.imread(imagepath)

img = cv2.resize(img, (224, 224))

orig = img.copy()

img = image.img_to_array(img)

img = np.expand_dims(img, axis=0)

img = img/255



print("[Info] predicting output")

prediction = model.predict(img)

if (prediction<0.5):

    print("def_front")

    cv2.putText(orig, "def_front", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

else:

    print("ok_front")

    cv2.putText(orig, "ok_front", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    

plt.imshow(orig)

plt.axis('off')

plt.show()
%matplotlib inline

imagepath = "../input/real-life-industrial-dataset-of-casting-product/casting_data/test/ok_front/cast_ok_0_1003.jpeg"

img = cv2.imread(imagepath)

img = cv2.resize(img, (224, 224))

orig = img.copy()

img = image.img_to_array(img)

img = np.expand_dims(img, axis=0)

img = img/255



print("[Info] predicting output")

prediction = model.predict(img)

if (prediction<0.5):

    print("def_front")

    cv2.putText(orig, "def_front", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

else:

    print("ok_front")

    cv2.putText(orig, "ok_front", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    

plt.imshow(orig)

plt.axis('off')

plt.show()
%matplotlib inline

imagepath = "../input/real-life-industrial-dataset-of-casting-product/casting_data/test/ok_front/cast_ok_0_1069.jpeg"

img = cv2.imread(imagepath)

img = cv2.resize(img, (224, 224))

orig = img.copy()

img = image.img_to_array(img)

img = np.expand_dims(img, axis=0)

img = img/255



print("[Info] predicting output")

prediction = model.predict(img)

if (prediction<0.5):

    print("def_front")

    cv2.putText(orig, "def_front", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

else:

    print("ok_front")

    cv2.putText(orig, "ok_front", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    

plt.imshow(orig)

plt.axis('off')

plt.show()