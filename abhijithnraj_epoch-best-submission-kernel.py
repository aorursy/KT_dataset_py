# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input/train data/Train data"))



# Any results you write to the current directory are saved as output.
!ls ../input/train\ data
import os

import cv2

import matplotlib.pyplot as plt

import numpy as np
trainl=os.listdir("../input/train data/Train data/leukocoria")

train_l=[]

print(trainl)

for i in trainl:

    im=cv2.imread("../input/train data/Train data/leukocoria/"+i)

    im=cv2.cvtColor(im,cv2.COLOR_BGR2RGB)

    im=cv2.resize(im,(150,150))

    train_l.append(im)

trainl=os.listdir("../input/train data/Train data/non-leukocoria")

train_nl=[]

for i in trainl:

    im=cv2.imread("../input/train data/Train data/non-leukocoria/"+i)

    im=cv2.cvtColor(im,cv2.COLOR_BGR2RGB)

    im=cv2.resize(im,(150,150))

    train_nl.append(im)



trainx=np.array(train_nl+train_l)

trainy=np.array([0 for i in range(len(train_nl))]+[1 for i in range(len(train_l))])



from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(trainx, trainy, test_size=0.10)
i=30

plt.imshow(train_l[i])

print(train_l[i].shape)

plt.title("Leukemia")

plt.show()

plt.imshow(train_nl[i])

print(train_nl[i].shape)

plt.title("Non Leukemia")

plt.show()
# Define model

from keras.models import Sequential

from keras.layers import Dense, Conv2D, Flatten,MaxPooling2D,Dropout

layer_name="Convolution"

def create_CNN_model():

    num_classes=2

    model=Sequential()

    model.add(Conv2D(3, kernel_size=(4, 4),

                     activation='relu',

                     input_shape=(150,150,3),name=layer_name+"0"))

    for i in range(9):

        model.add(Conv2D(30, (3, 3), activation='relu',name=layer_name+str(1+i)))

    model.add(Conv2D(3, (3, 3), activation='relu'))

    model.add(MaxPooling2D(pool_size=(3, 3)))

    model.add(Dropout(0.2))

    model.add(Conv2D(30, (3, 3), activation='relu'))

    model.add(MaxPooling2D(pool_size=(3, 3)))

    model.add(Dropout(0.2))

    model.add(Conv2D(60, (3, 3), activation='relu'))

    model.add(MaxPooling2D(pool_size=(3, 3)))

    model.add(Dropout(0.2))

    model.add(Flatten())

    model.add(Dense(120, activation='relu'))

    model.add(Dropout(0.2))

    model.add(Dense(120, activation='relu'))

    model.add(Dropout(0.2))

    model.add(Dense(2, activation='softmax'))

    return model

CNN_model=create_CNN_model()

CNN_model.summary()

from keras.utils import to_categorical

CNN_model.compile(optimizer="adam",

              loss='categorical_crossentropy',

              metrics=['acc'])

history = CNN_model.fit(X_train,to_categorical(y_train),

        epochs=100,validation_data=(X_test,to_categorical(y_test)),batch_size=50)
from sklearn.neighbors import KNeighborsClassifier

knn=KNeighborsClassifier(n_neighbors=3)

knn.fit([i.flatten() for i in trainx],trainy)
from sklearn.svm import SVC

svm=SVC(kernel="poly")

svm.fit([i.flatten() for i in X_train],y_train)
evalfolder="../input/evaluation data/Evaluation data"

evall=os.listdir(evalfolder)

evall=sorted(evall,key=lambda i:int(i.split(".")[0]))
eval_l=[]

for i in evall:

    im=cv2.imread("../input/evaluation data/Evaluation data/"+i)

    im=cv2.cvtColor(im,cv2.COLOR_BGR2RGB)

    im=cv2.resize(im,(150,150))

    eval_l.append(im)

eval_l=np.array(eval_l)

    
pred1=CNN_model.predict_classes(eval_l)
pred2=knn.predict([i.flatten() for i in eval_l])
pred3=svm.predict([i.flatten() for i in eval_l])
from collections import Counter

pred=[]

for i in range(len(eval_l)):

    p=[pred1[i],pred2[i],pred3[i]]

    pred.append(max(Counter(p)))

pred
import pandas as pd

m=pd.read_csv("../input/SampleSubmission.csv")

for i in range(len(m)):

    m.iloc[i][1]=pred[i]

m.to_csv("submisstion.csv",index=False)