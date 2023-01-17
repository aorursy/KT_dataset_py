#import library
#from scipy import ndimage
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import matplotlib.pyplot as plt
import cv2
from keras.models import Model
import os
import glob
import keras,os
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D , Flatten,Dropout,Activation
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from keras.utils import to_categorical

####read data and label
#readdddd
img_dir ="/kaggle/input/lfwlfw/lfw_funneled"
data_path = os.path.join(img_dir,'*')
##testttt

 

import numpy as np
from sklearn.model_selection import train_test_split

X = np.random.random((10000,70000))
Y = np.random.random((10000,))
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.33, random_state=42)



####files
files = glob.glob(data_path)
files=list(files)
data1 = []
####fffff
f=[]
label=[]

for i in files:
    for j in glob.glob(i+"/*.*"):
        
        label.append(i)

for f1 in glob.glob(data_path+"/*.*"):
    f.append(f1)
    img=cv2.imread(f1)
    img=cv2.resize(img,(224,224),interpolation=cv2.INTER_AREA)
    data1.append(img)
#make model
model = Sequential()
model.add(Conv2D(input_shape=(224,224,3),filters=64,kernel_size=(3,3),padding="same", activation="relu"))
model.add(Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))


model.add(Conv2D(filters=4096, kernel_size=(7,7), padding="same", activation="relu"))
model.add(Conv2D(filters=4096, kernel_size=(1,1), padding="same", activation="relu"))
model.add(Conv2D(filters=2622, kernel_size=(1,1), padding="same", activation="relu"))
model.add(Flatten())
model.add(Activation('softmax'))



##load model
from keras.models import model_from_json
model.load_weights('/kaggle/input/vggfaceweights/vgg_face_weights.h5')
###split data1
data1_train=data1[0:int(0.8*len(data1))]
data1_test=data1[0:int(0.8*len(data1)):]
label_train=label[0:int(0.7*len(label))]
label_test=label[0:int(0.7*len(label)):]

label_train=np.array(label_train)
label_train=np.reshape(label_train,(-1,1))
from sklearn import preprocessing
enc = preprocessing.OneHotEncoder()

files=np.array(files)
files=np.reshape(files,(-1,1))
enc.fit(files)

la_train = enc.transform(label_train).toarray()

#############################agaze model





#build model and feature
vgg_face=Model(inputs=model.layers[0].input,outputs=model.layers[-2].output)

feature=vgg_face.predict(np.array(data1))

#svm model
#split to test and train

X_train, X_test, y_train, y_test = train_test_split(feature, label, test_size = 0.3)

#make svm model
from sklearn import svm

svclassifier =svm.SVC(decision_function_shape='ovo')
svclassifier.fit(X_train, y_train)

y_pred = svclassifier.predict(X_test)


#evaluations

print('score', svclassifier.score(X_test, y_test))