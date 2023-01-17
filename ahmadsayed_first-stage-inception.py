import os

from tqdm import tqdm

import numpy as np

import cv2

from random import shuffle



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



# print(len(os.listdir(wrist_path_test)))

shoulder = 0

forearm = 1

hand = 2

finger = 3

humerus = 4

elbow = 5

wrist = 6



IMG_SIZE = 224
def craete_label(class_name):

    label = np.zeros(7)

    label[class_name] = 1

    return label



def create_train_data(train_data, path, bone_number):

    m = 0

    for item in tqdm(os.listdir(path)):

        patient_path = os.path.join(path, item)

        t = False

        for patient_study in os.listdir(patient_path): 

            p_path = os.path.join(patient_path, patient_study)

            label = craete_label(bone_number)

            if t == True:

                break

            for patient_image in os.listdir(p_path):

                if m >= 600:

                    t = True

                    break

                m += 1

                image_path = os.path.join(p_path, patient_image)

                bgr = cv2.imread(image_path)

                lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)

                lab_planes = cv2.split(lab)

                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

                lab_planes[0] = clahe.apply(lab_planes[0])

                lab = cv2.merge(lab_planes)

                bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

                img = cv2.resize(bgr, (IMG_SIZE, IMG_SIZE))

                img = np.divide(img, 255)

                train_data.append([np.array(img), label])

    shuffle(train_data)

    print("Done")
train_data = []

create_train_data(train_data, shoulder_path_train, shoulder)

print("Shoulder Data Number :", len(train_data))



create_train_data(train_data, forearm_path_train, forearm)

print("forearm Data Number :", len(train_data))



create_train_data(train_data, hand_path_train, hand)

print("hand Data Number :", len(train_data))



create_train_data(train_data, finger_path_train, finger)

print("finger Data Number :", len(train_data))



create_train_data(train_data, humerus_path_train, humerus)

print("humerus Data Number :", len(train_data))



create_train_data(train_data, elbow_path_train, elbow)

print("elbow Data Number :", len(train_data))



create_train_data(train_data, wrist_path_train, wrist)

print("wrist Data Number :", len(train_data))
X = np.array([i[0] for i in train_data]).reshape(-1, IMG_SIZE, IMG_SIZE, 3)

print("Train Image Load Succesfully")

print(X.shape)

y = np.array([i[1] for i in train_data])

print("Train Label Load Succeffully")

print(y.shape)
from keras import models

from keras import layers

from keras import optimizers

from keras.optimizers import SGD, Adam

from keras.applications import InceptionV3

Incp_con = InceptionV3(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))

Incp_con.summary()
model = models.Sequential()

model.add(Incp_con)
model.summary()

model.add(layers.Flatten())

model.add(layers.Dense(1024, activation='relu'))

model.add(layers.Dropout(0.5))

model.add(layers.Dense(7, activation='softmax'))

print("-----------------------------")

model.summary()
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)



model.compile( loss = "categorical_crossentropy", 

               optimizer = sgd, 

               metrics=['accuracy']

             )
model.fit(X, y, epochs=400, validation_split=0.2, batch_size=32)

train_data = []

create_train_data(train_data, shoulder_path_test, shoulder)

create_train_data(train_data, forearm_path_test, forearm)

create_train_data(train_data, hand_path_test, hand)

create_train_data(train_data, finger_path_test, finger)

create_train_data(train_data, humerus_path_test, humerus)

create_train_data(train_data, elbow_path_test, elbow)

create_train_data(train_data, wrist_path_test, wrist)
x_test = np.array([i[0] for i in train_data]).reshape(-1, IMG_SIZE, IMG_SIZE, 3)

print("Train Image Load Succesfully")

print(x_test.shape)

y_test = np.array([i[1] for i in train_data])

print("Train Label Load Succeffully")

print(y_test.shape)
result_v1 = model.predict_classes(x_test)
res = np.zeros(len(y_test))

for i in range(len(y_test)):

    res[i] = np.argmax(y_test[i])
count = 0

for i in range(len(res)):

    if int(res[i]) == result_v1[i] :

        count += 1

        

print("Test Accuracy : ", count / len(res) * 100 , "%")
model.save("model.h5")
model.evaluate(x_test, y_test)