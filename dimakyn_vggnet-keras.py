!pip install imutils
# import the necessary packages

from keras.models import Sequential 

from keras.layers.normalization import BatchNormalization

from keras.layers.convolutional import Conv2D

from keras.layers.convolutional import MaxPooling2D

from keras.layers.core import Activation

from keras.layers.core import Dense

from keras.layers.core import Flatten

from keras.layers.core import Dropout

from keras import backend as K 



from keras.preprocessing.image import ImageDataGenerator

from keras.optimizers import Adam

from keras.preprocessing.image import img_to_array

from sklearn.preprocessing import LabelBinarizer

from sklearn.model_selection import train_test_split

import matplotlib.pyplot  as plt 

from imutils import paths 

import imutils

import numpy as np 

import random 

import pickle 

import cv2 

import os 
class SmallVGGNet:

    @staticmethod

    def build(width, heigth, depth, classes):

        # initialize the model along with the input shape to be

        # "channels last" and the channels dimension itself

        model = Sequential()

        inputShape = (width, heigth, depth)

        chanDim = -1



        # if we are using "channels first", update the input shape and channels dimension

        if K.image_data_format() == "channels_first":

            inputShape = (depth, width, heigth)

            chanDim = 1



        # CONV => RELU => POOL

        model.add(Conv2D(32, (3,3), padding="same", input_shape=inputShape))

        model.add(Activation("relu"))

        model.add(BatchNormalization(axis=chanDim))

        model.add(MaxPooling2D(pool_size=(3,3)))

        model.add(Dropout(0.25))



        # (CONV => RELU) * 2 => POOL

        model.add(Conv2D(64, (3,3), padding="same"))

        model.add(Activation("relu"))

        model.add(BatchNormalization(axis=chanDim))

        model.add(Conv2D(64, (3,3), padding="same"))

        model.add(Activation("relu"))

        model.add(BatchNormalization(axis=chanDim))

        model.add(MaxPooling2D(pool_size=(2,2)))

        model.add(Dropout(0.25))



        # (CONV => RELU) * 2 => POOL

        model.add(Conv2D(128, (3,3), padding="same"))

        model.add(Activation("relu"))

        model.add(BatchNormalization(axis=chanDim))

        model.add(Conv2D(128, (3,3), padding="same"))

        model.add(Activation("relu"))

        model.add(BatchNormalization(axis=chanDim))

        model.add(MaxPooling2D(pool_size=(2,2)))

        model.add(Dropout(0.25))



        # first (and only) set of FC => RELU layers

        model.add(Flatten())

        model.add(Dense(1024))

        model.add(Activation("relu"))

        model.add(BatchNormalization())

        model.add(Dropout(0.5))



        # softmax classifier

        model.add(Dense(classes))

        model.add(Activation("softmax"))



        # return the constructed network architecture

        return model
EPOCHS = 10

INIT_LR = 1e-3

BS = 32

IMAGE_DIMS = (96, 96, 3)
data = []

labels = []
print("[INFO] loading images...")

imagePaths = sorted(list(paths.list_images("/kaggle/input/the-simpsons-characters-dataset/simpsons_dataset/simpsons_dataset/")))

random.seed(42)

random.shuffle(imagePaths)
for imagePath in imagePaths:

    # load the image, pre-process it, and store it in the data list

    image = cv2.imread(imagePath)

    image = cv2.resize(image, (IMAGE_DIMS[1], IMAGE_DIMS[0]))

    image = img_to_array(image)

    data.append(image)



    # extract the class label from the image path and update the labels list

    label = imagePath.split(os.path.sep)[-2]

    labels.append(label)
data = np.array(data, dtype="float") / 255.0

labels = np.array(labels)

print("[INFO] data matrix: {:.2f}MB".format(data.nbytes / (1024 * 1000.0)))
lb = LabelBinarizer()

labels = lb.fit_transform(labels)
(train_X, val_X, train_Y, val_Y) = train_test_split(data, labels, test_size=0.2, random_state=42)
aug = ImageDataGenerator(rotation_range=25, width_shift_range=0.1,

                         height_shift_range=0.1, shear_range=0.2,

                         horizontal_flip=True, fill_mode="nearest")
print("[INFO] compiling model...")

model = SmallVGGNet.build(width=IMAGE_DIMS[1], heigth=IMAGE_DIMS[0],

                          depth=IMAGE_DIMS[2], classes=len(lb.classes_))

opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)

model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
print("[INFO] training network...")

H = model.fit_generator(aug.flow(train_X, train_Y, batch_size=BS),

                        validation_data=(val_X, val_Y),

                        steps_per_epoch=len(train_X) // BS,

                        epochs=EPOCHS, verbose=1)
plt.style.use("ggplot")

plt.figure()

N = EPOCHS

plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")

plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")

plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")

plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")

plt.title("Training Loss and Accuracy")

plt.xlabel("Epoch #")

plt.ylabel("Loss/Accuracy")

plt.legend(loc="upper left")
def get_pred_res(file):

    image = cv2.imread(character_image)

    output = image.copy()

    #print(f"[INFO] character image:  {character_image.split(os.path.sep)[-1]}")

    

    # pre-process the image for classification

    image = cv2.resize(image, (96, 96))

    image = image.astype("float") / 255.0

    image = img_to_array(image)

    image = np.expand_dims(image, axis=0)

    

    # classify the input image

    #print("[INFO] classifying image...")

    proba = model.predict(image)[0]

    idx = np.argmax(proba)

    label = lb.classes_[idx]

    

    filename = "_".join(character_image.split(os.path.sep)[-1].split(".")[0].split("_")[:-1])

    correct = "correct" if filename == label else "incorrect"

    

    label = "{}: {:.2f}% ({})".format(label, proba[idx] * 100, correct)

    output = imutils.resize(output, width=400)



    return label, output

    #plt.imshow(output)
character_image = random.choice(list(paths.list_images("/kaggle/input/the-simpsons-characters-dataset/kaggle_simpson_testset/kaggle_simpson_testset")))

label, output = get_pred_res(character_image)

plt.title(label)

plt.imshow(output)