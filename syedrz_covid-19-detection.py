# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
!pip install imutils
# import the necessary libraries

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.applications import VGG16

from tensorflow.keras.layers import AveragePooling2D

from tensorflow.keras.layers import Dropout

from tensorflow.keras.layers import Flatten

from tensorflow.keras.layers import Dense

from tensorflow.keras.layers import Input

from tensorflow.keras.models import Model

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.utils import to_categorical

from sklearn.preprocessing import LabelBinarizer

from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix

from imutils import paths

import matplotlib.pyplot as plt

import numpy as np

import cv2

import os
# initialize the initial learning rate, number of epochs to train for,

# and batch size

INIT_LR = 1e-3

EPOCHS = 25

BS = 8



#The path to our input dataset of chest X-ray images.

dataset_dir = '/kaggle/input/'

plot_path = '/kaggle/working/plot.png'

model_path = '/kaggle/working/covid19.model'
# grab the list of images in our dataset directory, then initialize

# the list of data (i.e., images) and class images

print("[INFO] loading images...")

imagePaths = list(paths.list_images(dataset_dir))

data = []

labels = []

# loop over the image paths

for imagePath in imagePaths:

    # extract the class label from the filename

    label = imagePath.split(os.path.sep)[-2]

    # load the image, swap color channels, and resize it to be a fixed

    # 224x224 pixels while ignoring aspect ratio

    image = cv2.imread(imagePath)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image = cv2.resize(image, (224, 224))

    # update the data and labels lists, respectively

    data.append(image)

    labels.append(label)

data1 = data.copy()

labels1 = labels.copy()

# convert the data and labels to NumPy arrays while scaling the pixel

# intensities to the range [0, 255]

data = np.array(data) / 255.0

labels = np.array(labels)
# perform one-hot encoding on the labels

lb = LabelBinarizer()

labels = lb.fit_transform(labels)

labels = to_categorical(labels); print(labels)

# partition the data into training and testing splits using 80% of

# the data for training and the remaining 20% for testing

(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.20, stratify=labels, random_state=42)



# initialize the training data augmentation object

trainAug = ImageDataGenerator(rotation_range=15, fill_mode="nearest")
# load the VGG16 network, ensuring the head FC layer sets are left

# off

baseModel = VGG16(weights="imagenet", include_top=False, input_tensor=Input(shape=(224, 224, 3)))

# construct the head of the model that will be placed on top of the

# the base model

headModel = baseModel.output

headModel = AveragePooling2D(pool_size=(4, 4))(headModel)

headModel = Flatten(name="flatten")(headModel)

headModel = Dense(64, activation="relu")(headModel)

headModel = Dropout(0.5)(headModel)

headModel = Dense(2, activation="softmax")(headModel)

# place the head FC model on top of the base model (this will become

# the actual model we will train)

model = Model(inputs=baseModel.input, outputs=headModel)

# loop over all layers in the base model and freeze them so they will

# *not* be updated during the first training process

for layer in baseModel.layers:

    layer.trainable = False
# compile our model

print("[INFO] compiling model...")

opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)

model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

# train the head of the network

print("[INFO] training head...")

H = model.fit_generator(

    trainAug.flow(trainX, trainY, batch_size=BS),

    steps_per_epoch=len(trainX) // BS,

    validation_data=(testX, testY),

    validation_steps=len(testX) // BS,

    epochs=EPOCHS)
# make predictions on the testing set

print("[INFO] evaluating network...")

predIdxs = model.predict(testX, batch_size=BS)

# for each image in the testing set we need to find the index of the

# label with corresponding largest predicted probability

predIdxs = np.argmax(predIdxs, axis=1)

# show a nicely formatted classification report

print(classification_report(testY.argmax(axis=1), predIdxs, target_names=lb.classes_))
rows = 3

columns = 3

fig = plt.figure(figsize=(20, 20))

for m in range(1, 10):

    if str(predIdxs[m-1]) == "0":

        text = "NORMAL"

        color = (0, 255, 0)

    elif str(predIdxs[m-1]) == "1":

        text = "COVID"

        color = (255, 0, 0)

    img = testX[m-1].copy()

    # Window name in which image is displayed 

    window_name = text

  

    # font 

    font = cv2.FONT_HERSHEY_SIMPLEX 

  

    # org 

    org = (50, 50) 

  

    # fontScale 

    fontScale = 1

  

    # Line thickness of 2 px 

    thickness = 2

    img = cv2.putText(img, text, org, font,

                      fontScale, color, thickness, cv2.LINE_AA)

    fig.add_subplot(rows, columns, m)

    plt.imshow(img)

    plt.title("Pred: " + text)

    plt.axis('off')

plt.show()
rows = 3

columns = 3

fig = plt.figure(figsize=(20, 20))

for m in range(1, 10):

    if str(testY.argmax(axis=1)[m-1]) == "0":

        text = "NORMAL"

        color = (0, 255, 0)

    elif str(testY.argmax(axis=1)[m-1]) == "1":

        text = "COVID"

        color = (255, 0, 0)

    img = testX[m-1].copy()

    # Window name in which image is displayed 

    window_name = text

  

    # font 

    font = cv2.FONT_HERSHEY_SIMPLEX 

  

    # org 

    org = (50, 50) 

  

    # fontScale 

    fontScale = 1

  

    # Line thickness of 2 px 

    thickness = 2

    img = cv2.putText(img, text, org, font,

                      fontScale, color, thickness, cv2.LINE_AA)

    fig.add_subplot(rows, columns, m)

    plt.imshow(img)

    plt.title("Ground Truth: " + text)

    plt.axis('off')

plt.show()
# compute the confusion matrix and and use it to derive the raw

# accuracy, sensitivity, and specificity

cm = confusion_matrix(testY.argmax(axis=1), predIdxs)

total = sum(sum(cm))

acc = (cm[0, 0] + cm[1, 1]) / total

sensitivity = cm[0, 0] / (cm[0, 0] + cm[0, 1])

specificity = cm[1, 1] / (cm[1, 0] + cm[1, 1])

# show the confusion matrix, accuracy, sensitivity, and specificity

print(cm)

print("acc: {:.4f}".format(acc))

print("sensitivity: {:.4f}".format(sensitivity))

print("specificity: {:.4f}".format(specificity))
# plot the training loss and accuracy

N = EPOCHS

plt.style.use("ggplot")

plt.figure()

plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")

plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")

plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")

plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")

plt.title("Training Loss and Accuracy on COVID-19 Dataset")

plt.xlabel("Epoch #")

plt.ylabel("Loss/Accuracy")

plt.legend(loc="lower left")

plt.savefig(plot_path)



# serialize the model to disk

print("[INFO] saving COVID-19 detector model...")

model.save(model_path, save_format="h5")