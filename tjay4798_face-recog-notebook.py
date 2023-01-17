!pip install dlib
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import cv2

import os

print(os.listdir("../input"))

import glob



from scipy.spatial import distance as dist

import time

import dlib

from collections import OrderedDict

import h5py

import matplotlib.pyplot as plt

%matplotlib inline

# Any results you write to the current directory are saved as output.
import keras

# from keras.applications.nasnet import NASNetMobile, preprocess_input

from keras.applications.densenet import DenseNet201, preprocess_input

from keras.layers import *

from keras.models import *

from keras.callbacks import *

from keras.metrics import *

from keras.losses import *
#SPECIFICALLY TO CONVERT DLIB OBJ. TO NUMPY ARRAY

def shape_to_np(shape, dtype="int"):

    # initialize the list of (x, y)-coordinates

    coords = np.zeros((68, 2), dtype=dtype)

 

    # loop over the 68 facial landmarks and convert them

    # to a 2-tuple of (x, y)-coordinates

    for i in range(0, 68):

        coords[i] = (shape.part(i).x, shape.part(i).y)

 

    # return the list of (x, y)-coordinates

    return coords
def rect_to_bb(rect):

    # take a bounding predicted by dlib and convert it

    # to the format (x, y, w, h) as we would normally do

    # with OpenCV

    x = rect.left()

    y = rect.top()

    w = rect.right() - x

    h = rect.bottom() - y

 

    # return a tuple of (x, y, w, h)

    return (x, y, w, h)
detector = dlib.get_frontal_face_detector()

predictor = dlib.shape_predictor('/home/tejas/Desktop/CV/shape_predictor_68_face_landmarks.dat')
#having input size as (96,96)

inputs = Input(shape=(96,96,3))
#Densenet201

base_model = DenseNet201(weights = 'imagenet',include_top=False,input_shape=(96,96,3))
ip = base_model(inputs)

ip1 = GlobalMaxPooling2D()(ip)

ip2 = GlobalAveragePooling2D()(ip)

ip3 = Flatten()(ip)

op = Concatenate(axis=-1)([ip1,ip2,ip3])



#128bit vector encodings

op = Dense(128, activation='relu')(op)

norm_layer = Lambda(lambda  x: K.l2_normalize(x, axis=1), name='norm_layer')(op)

model = Model(inputs, norm_layer)

# model.compile(optimizer=Adam(0.0001), loss=binary_crossentropy, metrics=['rmse'])

# model.summary()
# model_face_json = model.to_json()



# with open('model_face.json','w') as json_file:

#     json_file.write(model_face_json)

    

# model.save_weights('model_face_wts.h5')
# Saving the model

model.save('model_face_recog.h5',)
# empty dictionary for saving face encodings of people ({'name_of_person':[encodings(128-long)]})

encodings = dict()
#load the model before

# model = load_model('.......')
#Take face encodings from a few different images of the same person and average it and save it in the dictionary

encodings_tejas = []

tejas_images = glob.glob('../input/tejas/Tejas/*')

for path in tejas_images:

    photo = cv2.imread(path,cv2.COLOR_BGR2GRAY)

    photo = cv2.resize(photo, (96,96))

    photo = np.reshape(photo, (1,96,96,3))

    encodings_tejas.append(model.predict(photo))

    

encodings_tejas_final = np.mean(np.array(encodings_tejas),axis=0)



encodings['tejas'] = encodings_tejas_final
#To recognize faces, get the encodings of the detected face and get euclidean_distance of the embeddings with each of our saved face-embeddings.

#The closest one to the detected embeddings will be displayed



#If the minimum distance is greater than 0.7 then that face does not match any of our given faces and should be labeled unknown



#The threshold can be adjusted according to our needs
def get_face(photo,i):

    min_dist = 200

    thresh = 0.70

    for name, embed in encodings.items():

        euclidean_distance = np.linalg.norm(encodings[name] - model.predict(photo))

        if euclidean_distance < thresh:

            min_dist = euclidean_distance

            key = name

            

    if min_dist >= thresh:

        print('Euclidean distance : {}'.format(min_dist))

        return "Unknown"

    else:

        print('Euclidean distance : {}'.format(min_dist))

        cv2.imwrite('/home/abc/xyx/Unknown'+str(i)+'.jpg',photo)

        return name

        
############################ Testing ################################
tej = cv2.imread('../input/tejas/Tejas/tejas3.jpg',cv2.COLOR_BGR2GRAY)

tej = cv2.resize(tej, (96,96))

tej = np.reshape(tej, (1,96,96,3))
# euclidean_distance = np.linalg.norm(encodings['tejas'] - model.predict(tej))

# print (euclidean_distance)

get_face(tej)
abhi = cv2.imread('../input/abhilash/Abhilash/abhilash2.jpg',cv2.COLOR_BGR2GRAY)

abhi = cv2.resize(abhi, (96,96))

abhi = np.reshape(abhi, (1,96,96,3))
# euclidean_distance = np.linalg.norm(encodings['tejas'] - model.predict(abhi))

# print (euclidean_distance)

get_face(abhi)
################################## Testing over ##############################################################
#USE THIS CODE TO SAVE ENCODING EVERY TIME THE ENCODINGS DICTIONARY IS UPDATED



# import pickle

# f = open("encodings.pkl","wb")

# pickle.dump(encodings,f)

# f.close()
cap = cv2.VideoCapture(0)



while True: 

    ret, frame = cap.read() 

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 

    rects = detector(gray, 0)



    for i,rect in enumerate(rects):

        shape = predictor(gray,rect)

        shape = shape_to_np(shape)



        (x,y,w,h) = rect_to_bb(rect)

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        frame_roi = frame[y:y+h, x:x+w]

        frame_roi = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        pred_roi = cv2.resize(frame_roi, (96,96))

        #pred_roi = np.array(pred_roi)

        pred_roi = np.reshape(pred_roi,(1,96,96,3))

        #pred = model.predict(pred_roi)

        id_ = get_face(pred_roi)

        cv2.putText(frame, "Face #{} {}".format((i + 1), id_), (x - 10, y - 10),

        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 3)



    frame = cv2.resize(frame, (1200,900))

    cv2.imshow('frame', frame)

    

    #press 1 and q keys to exit

    if cv2.waitKey(1) & 0xFF == ord('q'):

        break



cap.release() 

cv2.destroyAllWindows()
