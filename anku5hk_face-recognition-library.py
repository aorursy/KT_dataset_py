# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
!pip3 install face_recognition
import dlib

import face_recognition

import os

import re

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
ROOT = '../input/5-faces-dataset/Five_Faces/'
# print number of images per class



flds = [ROOT+f for f in os.listdir(ROOT)]

print('Number of images per subject')

print('Bill Gates',len(os.listdir(flds[0])))

print('Jack Ma',len(os.listdir(flds[1])))

print('Narendra Modi',len(os.listdir(flds[2])))

print('Donald Trump',len(os.listdir(flds[3])))

print('Elon Musk',len(os.listdir(flds[4])))
face1 = face_recognition.load_image_file(ROOT + 'jack/jack10.jpg')

face2 = face_recognition.load_image_file(ROOT + 'gates/gates1.jpg')

face3 = face_recognition.load_image_file(ROOT + 'jack/jack1.jpg')



face_encd1 = face_recognition.face_encodings(face1)[0]

face_encd2 = face_recognition.face_encodings(face2)[0]

face_encd3 = face_recognition.face_encodings(face3)[0]



faces = {'Encoding1': face1, 'Encoding2': face2, 'Predited': face3}

encd = face_recognition.compare_faces([face_encd1, face_encd2], face_encd3)

f, axs = plt.subplots(1,3)

plt.tight_layout()

for i, key in enumerate(faces.keys()):

  axs[i].imshow(faces[key])

  axs[i].set_title(key)

  if i ==2: 

     axs[i].set_title(encd)
# create 5 encodings(1 per subject) and compare with random faces from dataset



face1 = face_recognition.load_image_file(ROOT +'jack/jack10.jpg')

face2 = face_recognition.load_image_file(ROOT +'gates/gates1.jpg')

face3 = face_recognition.load_image_file(ROOT +'modi/modi103.jpg')

face4 = face_recognition.load_image_file(ROOT +'musk/musk104.jpg')

face5 = face_recognition.load_image_file(ROOT +'trump/donald trump speech106.jpg')



random_face1 = face_recognition.load_image_file(ROOT +'jack/jack109.jpg')

random_face2 = face_recognition.load_image_file(ROOT +'gates/gates123.jpg')

random_face3 = face_recognition.load_image_file(ROOT +'musk/musk121.jpg')



face_encd1 = face_recognition.face_encodings(face1)[0]

face_encd2 = face_recognition.face_encodings(face2)[0]

face_encd3 = face_recognition.face_encodings(face3)[0]

face_encd4 = face_recognition.face_encodings(face4)[0]

face_encd5 = face_recognition.face_encodings(face5)[0]



rnd_encd1 = face_recognition.face_encodings(random_face1)[0]

rnd_encd2 = face_recognition.face_encodings(random_face2)[0]

rnd_encd3 = face_recognition.face_encodings(random_face3)[0]



faces = [face_encd1,face_encd2,face_encd3,face_encd4,face_encd5]



encd1 = face_recognition.compare_faces(faces, rnd_encd1)

encd2 = face_recognition.compare_faces(faces, rnd_encd2)

encd3 = face_recognition.compare_faces(faces, rnd_encd3)



print('actual 1',encd1)

print('actual 2',encd2)

print('actual 4',encd3)
# create face embeddings for all images



embeddings = [] # store all embeddings

cl_lm = [] # per class number of embeddings detected



for fld in flds:

  for img in os.listdir(fld):

      try:

        image = face_recognition.load_image_file(fld+'/'+img)

        face_encodings = face_recognition.face_encodings(image)[0] # to 128 encodings of single face

      except:

        continue

      embeddings.append(face_encodings)

  print(len(embeddings)) # print to indexes of number of images, how many faces we added per class

  cl_lm.append(len(embeddings)) # make note for making Y
# make y using cl_lm



Y = np.zeros(len(embeddings))

Y[:cl_lm[0]] = 1

Y[cl_lm[0]:cl_lm[1]] = 2

Y[cl_lm[1]:cl_lm[2]] = 3

Y[cl_lm[2]:cl_lm[3]] = 4

Y[cl_lm[3]:cl_lm[4]] = 5
# number of embeddings



print(len(embeddings))

print(len(Y)) 
# split the data



X_train, X_test, Y_train, Y_test = train_test_split(embeddings, Y, test_size=0.2)
# predict using all embeddings(~700)



correct = 0

yhat = 0



def predict_results(result):

  ids = [i for i, val in enumerate(result) if val] 

  values, counts = np.unique(Y[ids], return_counts=True)

  yhat = values[np.argmax(counts)]

  return yhat



for test_embedding, y in zip(X_test, Y_test):



  result = face_recognition.compare_faces(embeddings, test_embedding) # a big bool array of 710

  yhat = predict_results(result)

  if yhat == y:

    correct+=1

print('Using all embeddings')

print('{0} correct out of {1}'.format(correct, len(Y_test)))
len(faces) # reminder
# predict with only 5 embeddings



correct = 0

yhat = 0



for test_embedding, y in zip(X_test, Y_test):



  result = face_recognition.compare_faces(faces, test_embedding)

  yhat = [i for i,val in enumerate(result) if val]

  if yhat:

    if yhat == y:

      correct+=1

print('using 5 faces embedding')

print('{0} correct out of {1}'.format(correct, len(Y_test)))