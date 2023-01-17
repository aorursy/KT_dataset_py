import os

from tqdm import tqdm

import numpy as np

import cv2

from random import shuffle

from random import shuffle

import keras

from keras.utils import to_categorical

from keras import models

from keras import layers

from keras import optimizers

from keras.optimizers import SGD, Adam

from keras.applications import InceptionV3

shoulder = 0

forearm = 1

hand = 2

finger = 3

humerus = 4

elbow = 5

wrist = 6



#SHOULDER ----> 0

shoulder_path_train = "../input/xr_shoulder_train/XR_SHOULDER_TRAIN"

shoulder_path_test = "../input/xr_shoulder_valid/XR_SHOULDER_VALID"



#FOREARM ----> 1

forearm_path_train = "../input/xr_forearm_train/XR_FOREARM_TRAIN"

forearm_path_test = "../input/xr_forearm_valid/XR_FOREARM_VALID"



#HAND  ----> 2

hand_path_train = "../input/xr_hand_train/XR_HAND_TRAIN"

hand_path_test = "../input/xr_hand_valid/XR_HAND_VALID"



#FINGER  ----> 3

finger_path_train = "../input/xr_finger_train/XR_FINGER_TRAIN"

finger_path_test = "../input/xr_finger_valid/XR_FINGER_VALID"



#HUMERUS ----> 4

humerus_path_train = "../input/xr_humerus_train/XR_HUMERUS_TRAIN"

humerus_path_test = "../input/xr_humerus_valid/XR_HUMERUS_VALID"



#ELBOW -----> 5

elbow_path_train = "../input/xr_elbow_train/XR_ELBOW_TRAIN"

elbow_path_test = "../input/xr_elbow_valid/XR_ELBOW_VALID"



#WRIST  -----> 6

wrist_path_train = "../input/xr_wrist_train/XR_WRIST_TRAIN"

wrist_path_test = "../input/xr_wrist_valid/XR_WRIST_VALID"
import os

print(os.listdir("../input"))
import os 

names = []

f = open("../input/train_image_paths.csv")

for row in f:

    names.append(row.strip())
names
train_paths = []

for name in names:

    label = None

    arr = name.split("/")

    if arr[2] == "XR_SHOULDER":

        root = shoulder_path_train

        label = shoulder

        

    elif arr[2] == "XR_FINGER":

        root = finger_path_train

        label = finger

        

    elif arr[2] == "XR_ELBOW":

        root = elbow_path_train

        label = elbow

        

    elif arr[2] == "XR_WRIST":

        root = wrist_path_train

        label = wrist

        

    elif arr[2] == "XR_HUMERUS":

        root = humerus_path_train

        label = humerus

        

    elif arr[2] == "XR_FOREARM":

        root = forearm_path_train

        label = forearm

        

    elif arr[2] == "XR_HAND":

        root = hand_path_train

        label = hand

        

    root  = root + "/" + arr[3] + "/" + arr[4] + "/" + arr[5]

    train_paths.append([root, label])
train_paths
import os 

valid_names = []

f = open("../input/valid_image_paths.csv")

for row in f:

    valid_names.append(row.strip())
valid_paths = []

for name in valid_names:

    label = None

    arr = name.split("/")

    if arr[2] == "XR_SHOULDER":

        root = shoulder_path_test

        label = shoulder

        

    elif arr[2] == "XR_FINGER":

        root = finger_path_test

        label = finger

        

    elif arr[2] == "XR_ELBOW":

        root = elbow_path_test

        label = elbow

        

    elif arr[2] == "XR_WRIST":

        root = wrist_path_test

        label = wrist

        

    elif arr[2] == "XR_HUMERUS":

        root = humerus_path_test

        label = humerus

        

    elif arr[2] == "XR_FOREARM":

        root = forearm_path_test

        label = forearm

        

    elif arr[2] == "XR_HAND":

        root = hand_path_test

        label = hand

        

    root  = root + "/" + arr[3] + "/" + arr[4] + "/" + arr[5]

    valid_paths.append([root, label])
train_paths
shuffle(valid_paths)

shuffle(train_paths)


def getImageArr(path, size):



    try:



        bgr = cv2.imread(path)

        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)

        lab_planes = cv2.split(lab)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

        lab_planes[0] = clahe.apply(lab_planes[0])

        lab = cv2.merge(lab_planes)

        bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        img = cv2.resize(bgr, (size, size))

        img = np.divide(img, 255)

        return img

    except Exception as e:

        img = np.zeros((size, size, 3))

        return img

import itertools

def imageSegmentationGenerator(pathes_labels, batch_size, output_size):

    

    x1 = np.array([i[0] for i in pathes_labels])

    y1 = np.array([i[1] for i in pathes_labels])

    zipped = itertools.cycle(zip(x1, y1))



    while True:

        X = []

        Y = []

        for _ in range(batch_size):

            pa, la = next(zipped)

            X.append(getImageArr(pa, 224))

            Y.append(to_categorical(la, 7))



        yield np.array(X), np.array(Y)
Incp_con = InceptionV3(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

Incp_con.summary()
model = models.Sequential()

model.add(Incp_con)
model.add(layers.Flatten())

model.add(layers.Dense(1024, activation='relu'))

model.add(layers.Dropout(0.5))

model.add(layers.Dense(7, activation='softmax'))



model.summary()
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)



model.compile( loss = "categorical_crossentropy", 

               optimizer = sgd, 

               metrics=['accuracy']

             )
G = imageSegmentationGenerator(train_paths, 32, 224)
G
G2 = imageSegmentationGenerator(valid_paths, 32, 224)
epochs = 10
epochs
for ep in range(epochs):

    model.fit_generator(G, 512, validation_data=G2, validation_steps=200,  epochs=10)

    model.save_weights("wight" + "." + str(ep))

    model.save("mo" + ".model." + str(ep))
model.save("model.h5")