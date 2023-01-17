import os

from tqdm import tqdm

import numpy as np

import cv2

from random import shuffle

import keras

from keras.layers import Dense

from keras.models import Sequential

from keras.optimizers import Adam, SGD

from keras.layers import Conv2D, Activation, ZeroPadding2D, MaxPooling2D, Flatten, Dropout

from  keras.models import load_model

model_path="../input/s-1-m/Stage_One_50_epochs.h5"



#SHOULDER ----> 0

shoulder_path_test = "../input/ourgreatestdream/xr_shoulder_valid/XR_SHOULDER_VALID"



#FOREARM ----> 1

forearm_path_test = "../input/ourgreatestdream/xr_forearm_valid/XR_FOREARM_VALID"



#HAND  ----> 2

hand_path_test = "../input/ourgreatestdream/xr_hand_valid/XR_HAND_VALID"



#FINGER  ----> 3

finger_path_test = "../input/ourgreatestdream/xr_finger_valid/XR_FINGER_VALID"



#HUMERUS ----> 4

humerus_path_test = "../input/ourgreatestdream/xr_humerus_valid/XR_HUMERUS_VALID"



#ELBOW -----> 5

elbow_path_test = "../input/ourgreatestdream/xr_elbow_valid/XR_ELBOW_VALID"



#WRIST  -----> 6

wrist_path_test = "../input/ourgreatestdream/xr_wrist_valid/XR_WRIST_VALID"

from subprocess import check_output

print(check_output(["ls","../input"]).decode("utf8"))
shoulder = 0

forearm = 1

hand = 2

finger = 3

humerus = 4

elbow = 5

wrist = 6

postive = 1

negative = 0

size_before=[]

IMG_SIZE = 224
def craete_label(class_name):

    label = np.zeros(7)

    label[class_name] = 1

    return label



def create_label_2(class_name):

    label = np.zeros(2)

    label[class_name] = 1

    return label



def create_train_data(train_data, path, bone_number):

    take_part_of_data = int(len(os.listdir(path)) / 3)

    #print(take_part_of_data)

    m = 0

    cnt=0

    for item in tqdm(os.listdir(path)):

        m += 1

        if m == take_part_of_data:

            break

        patient_path = os.path.join(path, item)

        for patient_study in os.listdir(patient_path): 

            p_path = os.path.join(patient_path, patient_study)

            label = craete_label(bone_number)

            

            type_of_study = patient_study.split('_')[1]

            

            if type_of_study == "positive":

                class_con = postive

            else:

                class_con = negative

                

            label_abnorm=create_label_2(class_con)

#             print(label)

            for patient_image in os.listdir(p_path):

                cnt+=1

                image_path = os.path.join(p_path, patient_image)

                img = cv2.imread(image_path, 0)

                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)) 

                img = clahe.apply(img)

                img = np.divide(img, 255)

                train_data.append([np.array(img),label,label_abnorm])

    size_before.append(cnt)

    shuffle(train_data)

    print("Done")
test_data = []





create_train_data(test_data, shoulder_path_test, shoulder)

create_train_data(test_data, forearm_path_test, forearm)

create_train_data(test_data, hand_path_test, hand)

create_train_data(test_data, finger_path_test, finger)

create_train_data(test_data, humerus_path_test, humerus)

create_train_data(test_data, elbow_path_test, elbow)

create_train_data(test_data, wrist_path_test, wrist)

print("Finished")
print(len(test_data[0]))

x_test = np.array([i[0] for i in test_data]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)

print("Train Image Load Succesfully")

print(x_test.shape)

y_test = np.array([i[1] for i in test_data])

print("Train Label Load Succeffully")

print(y_test.shape)

z_test = np.array([i[2] for i in test_data])

print("Train Label for Abonrmality  Load Succeffully")

print(z_test.shape)
model=load_model(model_path)

result_v1 = model.predict_classes(x_test)
res = np.zeros(len(y_test))

for i in range(len(y_test)):

    res[i] = np.argmax(y_test[i])
count = 0

for i in range(len(res)):

    if int(res[i]) == result_v1[i] :

        count += 1

        

print("Test Accuracy : ", count / len(res) * 100 , "%")
x_test_shoulder,y_test_shoulder=[],[]

x_test_forearm,y_test_forearm=[],[]

x_test_hand,y_test_hand=[],[]

x_test_finger,y_test_finger=[],[]

x_test_humerus,y_test_humerus=[],[]

x_test_elbow,y_test_elbow=[],[]

x_test_wrist,y_test_wrist=[],[]

for i in range(len(result_v1)):

    if result_v1[i]==0:

       x_test_shoulder.append(x_test[i])

       y_test_shoulder.append(z_test[i])

    if result_v1[i]==1:

       x_test_forearm.append(x_test[i])

       y_test_forearm.append(z_test[i])

    if result_v1[i]==2:

       x_test_hand.append(x_test[i])

       y_test_hand.append(z_test[i])

    if result_v1[i]==3:

       x_test_finger.append(x_test[i])

       y_test_finger.append(z_test[i])

    if result_v1[i]==4:

       x_test_humerus.append(x_test[i])

       y_test_humerus.append(z_test[i])

    if result_v1[i]==5:

       x_test_elbow.append(x_test[i])

       y_test_elbow.append(z_test[i])

    if result_v1[i]==6:

       x_test_wrist.append(x_test[i])

       y_test_wrist.append(z_test[i])

        

x_test_shoulder,y_test_shoulder=np.array(x_test_shoulder),np.array(y_test_shoulder)

x_test_forearm,y_test_forearm=np.array(x_test_forearm),np.array(y_test_forearm)

x_test_hand,y_test_hand=np.array(x_test_hand),np.array(y_test_hand)

x_test_finger,y_test_finger=np.array(x_test_finger),np.array(y_test_finger)

x_test_humerus,y_test_humerus=np.array(x_test_humerus),np.array(y_test_humerus)

x_test_elbow,y_test_elbow=np.array(x_test_elbow),np.array(y_test_elbow)

x_test_wrist,y_test_wrist=np.array(x_test_wrist),np.array(y_test_wrist)

print("************Size of 1/3 of Bone Before classification*********")

print("Bone Shoulder : ",size_before[0])

print("Bone Forearm : ",size_before[1])

print("Bone2 Hand: ",size_before[2])

print("Bone3 Finger: ",size_before[3])

print("Bone4 Humerus: ",size_before[4])

print("Bone5 Elbow: ",size_before[5])

print("Bone6 Wrist: ",size_before[6])

#After

print("************Size of 1/3 of Bone After classification*********")

print("Bone Shoulder: ",len(x_test_shoulder))

print("Bone Forearm : ",x_test_forearm.shape[0])

print("Bone Hand : ",x_test_hand.shape[0])

print("Bone Finger : ",x_test_finger.shape[0])

print("Bone Humerus : ",x_test_humerus.shape[0])

print("Bone Elbow : ",x_test_elbow.shape[0])

print("Bone Wrist : ",x_test_wrist.shape[0])

sz=len(res)

print("Correct Data",count)

print("All Data ",sz)
def pred(path,x_test,y_test):

    model=load_model(path)

    result_v1 = model.predict_classes(x_test)

    res = np.zeros(len(y_test))

    for i in range(len(y_test)):

        res[i] = np.argmax(y_test[i])

    count = 0

    for i in range(len(res)):

       if int(res[i]) == result_v1[i] :

          count += 1

    return count
tot=0

'''model="../input/1-3elbowvgg150epochs/VGG_ELBOW150epoch.h5"

tot+=pred(model,x_test_elbow,y_test_elbow)

print("Done ",tot)

model="../input/1-3fingervgg150epochs/VGG_Finger150epoch.h5"

tot+=pred(model,x_test_finger,y_test_finger)

print("Done ",tot)

model="../input/1-3forearmvgg150epochs/VGG_FOREARM150epoch.h5"

tot+=pred(model,x_test_forearm,y_test_forearm)

print("Done ",tot)

model="../input/1-3handvgg150epochs/VGG_HAND150epoch.h5"

tot+=pred(model,x_test_hand,y_test_hand)

print("Done ",tot)

'''

# 322 already computed (small memory)

tot=322

model="../input/1-3humerusvgg150epochs/VGG_HUMERUS150epoch.h5"

tot+=pred(model,x_test_humerus,y_test_humerus)

print("Done")

model="../input/1-3shouldervgg150epochs/VGG_SHOULDER150epoch.h5"

tot+=pred(model,x_test_shoulder,y_test_shoulder)

print("Done")

model="../input/1-3wristvgg150epochs/VGG_WRIST150epoch.h5"

tot+=pred(model,x_test_wrist,y_test_wrist)

print("Done ",tot)



print("Test Accuracy for 2 Stages : ", tot / sz * 100 , "%")








