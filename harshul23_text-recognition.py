# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input/image1"))

# Any results you write to the current directory are saved as output.

from keras.models import Sequential
from keras.layers import Convolution2D,BatchNormalization
from keras.layers import MaxPooling2D,Dropout
from keras.layers import Flatten
from keras.layers import Dense
import cv2
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split

# Initialising the CNN
classifier = Sequential()
# Step 1 - Convolution
classifier.add(Convolution2D(32, (3, 3), input_shape = (50, 50,1), activation = 'relu',padding='same'))
# Adding a second convolutional layer
#classifier.add(BatchNormalization(axis=1))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(Dropout(0.25))


classifier.add(Convolution2D(64, (3, 3), activation = 'relu',padding='same'))
# Adding a second convolutional layer
classifier.add(Convolution2D(64, (3, 3), activation = 'relu',padding='same'))
#classifier.add(BatchNormalization(axis=1))

classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(Dropout(0.25))


classifier.add(Convolution2D(128, (3, 3), activation = 'relu',padding='same'))
# Adding a second convolutional layer
classifier.add(Convolution2D(128, (3, 3), activation = 'relu',padding='same'))
#classifier.add(BatchNormalization(axis=1))

classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(Dropout(0.25))
# Step 3 - Flattening
classifier.add(Flatten())
classifier.add(Dense(512,activation='relu'))
classifier.add(BatchNormalization())
classifier.add(Dropout(0.5))

# Step 4 - Full connection
classifier.add(Dense(output_dim = 62, activation = 'softmax'))

# Compiling the CNN

classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])


classifier.summary()


from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   zoom_range = 0.2,
                                   shear_range=0.2,

                                  )

training_set = train_datagen.flow_from_directory('../input/images/bmp/Bmp',
                                                target_size = (50,50),
                                               batch_size = 128,
                                                 color_mode= "grayscale",
                                              class_mode = 'categorical')


classifier.fit_generator(training_set,
                         epochs = 50,
                         steps_per_epoch=40,
                         )

import cv2
from keras.preprocessing.image import img_to_array
import numpy as np
img=cv2.imread('../input/quotes/down.jpeg')
imagee=img.copy()
img=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)

ret,img=cv2.threshold(img,180,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

k=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))

img = cv2.morphologyEx(img, cv2.MORPH_OPEN, k)
k1=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(1,1))
img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, k)

img1=cv2.Canny(img,0,255,2)

contours, hierarchy = cv2.findContours(img1,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

img2=cv2.cvtColor(img1,cv2.COLOR_GRAY2BGR)
font = cv2.FONT_HERSHEY_SIMPLEX

img3=cv2.drawContours(img2, contours, -1, (0,255,255), 1)
image=[]
p=[]
list=['0','1','2','3','4','5','6','7','8','9']
for i in range(65,91):
    list.append(chr(i))
for i in range(97,123):
    list.append(chr(i))
print(list)
for c in contours:
    x,y,w,h=cv2.boundingRect(c)
    if w>5 and h>5:
        
        img4=cv2.rectangle(img3,(x,y),(x+w,y+h),(255,255,0),1)     
        
        i=img3[y:y+h,x:x+w]
        i=cv2.resize(i,(50,50))
        i=cv2.cvtColor(i,cv2.COLOR_BGR2GRAY)
        i = i.astype("float") / 255.0
        
        ima = img_to_array(i)
        ima = np.expand_dims(ima, axis=0)
        pred = classifier.predict(ima)[0]
        #print(list[pred.argmax()],pred.max())
        img5=cv2.putText(img4,list[pred.argmax()],(x,y+h+20), font, 0.8 ,(255,255,255),2,cv2.LINE_AA)   
        p.append(list[pred.argmax()])
'''for c in contours:
    x,y,w,h=cv2.boundingRect(c)
    if w>5 and h>5:
        img5=cv2.putText(img4,list[pred.argmax()],(x,y+h-35), font, 0.5,(255,255,255),2,cv2.LINE_AA)        
from PIL import Image, ImageTk 
#img5.show()'''
import matplotlib.pyplot as plt
plt.imshow(img5)
   
plt.imshow(img)

'''from keras.preprocessing.image import img_to_array
import numpy as np 
list=['s1','s2']
image = cv2.imread('../input/picture/main-qimg-03de963368e748a6fb7e399772b09c48-c')
print(type(image))
# pre-process the image for classification
image = cv2.resize(image, (50,50))
ima=image

image = image.astype("float") / 255.0
image = img_to_array(image)
image = np.expand_dims(image, axis=0)
print(image.shape)

pred = classifier.predict(image)[0]
for i in range(2):
    if pred[i]>0.5:
        print(list[i],(pred[i]).astype('float32'))
    

print(pred)

#classifier.save('../input/model.h5')'''
'''import cv2
print((ima.shape))

gray=cv2.cvtColor(ima,cv2.COLOR_BGR2GRAY)
face_cascade = cv2.CascadeClassifier('../input/repository/informramiz-opencv-face-recognition-python-0edc6e0/opencv-files/haarcascade_frontalface_alt.xml')
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)
(x,y,w,h)=faces[0]
imm=gray[y:y+w, x:x+h]
print((gray.shape))
import matplotlib.pyplot as mat
mat.plot(imm)
cv2.waitKey()'''
'''import cv2
f=[]
l=[]
dir=sorted(list(os.listdir('../input/repository/informramiz-opencv-face-recognition-python-0edc6e0/training-data')))
for i in dir:
    s=sorted(list(os.listdir('../input/repository/informramiz-opencv-face-recognition-python-0edc6e0/training-data/'+i)))
    for j in s:
        print(i)
        k=cv2.imread('../input/repository/informramiz-opencv-face-recognition-python-0edc6e0/training-data/'+i+'/'+j)
        gray=cv2.cvtColor(k,cv2.COLOR_BGR2GRAY)
        face_cascade=cv2.CascadeClassifier('../input/repository/informramiz-opencv-face-recognition-python-0edc6e0/opencv-files/lbpcascade_frontalface.xml')
        faces=face_cascade.detectMultiScale(gray,1.2,9)
        if len(faces)>0:
            (x,y,w,h)=faces[0]
            face=gray[y:y+h,x:x+w]
            f.append(face)
            label=i
            if label=='s1':
                l.append(1)
            else:
                l.append(0)'''
'''face_rec=cv2.face.LBPHFaceRecognizer_create()
face_rec.train(f,np.array(l))
testimg='../input/repository/informramiz-opencv-face-recognition-python-0edc6e0/test-data/test1.jpg'
lab,conf=face_rec.predict(testimg)'''