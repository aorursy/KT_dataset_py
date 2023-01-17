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

        

        break

# Any results you write to the current directory are saved as output.
from sklearn.model_selection import train_test_split as tts

import cv2

import matplotlib.pyplot as plt

import sklearn

import random
path="/kaggle/input/DataSet/Train Images/Train Images"
df=pd.read_csv("/kaggle/input/DataSet/train.csv")
df.head()
labels={"Large":1,"Small":0}

dataset=[]

for label in os.listdir("/kaggle/input/DataSet/Train Images/Train Images"):

    path1=os.path.join(path,label)

    for images in os.listdir(os.path.join(path,label)):

        img=cv2.imread(os.path.join(path1,images),0)

        img=cv2.resize(img,(200,200))

        dataset.append([img,labels[label]])

        

        

        

        

print(os.path.join(path1,images))

plt.imshow(img)
X,y=[],[]

random.shuffle(dataset)

for data,clas in dataset:

    X.append(data)

    y.append(clas)

    
X=np.array(X)

y=np.array(y)

X=X/255
X=np.expand_dims(X,axis=4)

print(X.shape)
import keras
y=keras.utils.to_categorical(y, 2)
Xt,xt,yt,y_test=tts(X,y)
Xt.shape
from keras.layers import Dense,Conv2D,AveragePooling2D,MaxPooling2D,Flatten,Dropout

from keras import Sequential
model = Sequential()

 

#1st convolution layer

model.add(Conv2D(64, (5, 5), activation='relu', input_shape=(200,200,1)))

model.add(MaxPooling2D(pool_size=(3,3)))

 

#2nd convolution layer

model.add(Conv2D(64, (3, 3), activation='relu'))

model.add(Conv2D(64, (3, 3), activation='relu'))

model.add(AveragePooling2D(pool_size=(3,3)))

 

#3rd convolution layer

model.add(Conv2D(128, (3, 3), activation='relu'))

model.add(Conv2D(128, (3, 3), activation='relu'))

model.add(AveragePooling2D(pool_size=(2,2)))

 

model.add(Flatten())

 

#fully connected neural networks

model.add(Dense(1024, activation='relu'))

model.add(Dropout(0.2))

model.add(Dense(1024, activation='relu'))

model.add(Dropout(0.2))

 

model.add(Dense(2, activation='softmax'))
from keras.preprocessing.image import ImageDataGenerator

import keras
gen = ImageDataGenerator()

train_generator = gen.flow(Xt, yt, batch_size=32)



model.compile(loss='categorical_crossentropy'

, optimizer=keras.optimizers.Adam()

, metrics=['accuracy']

)

 

model.fit_generator(train_generator, steps_per_epoch=32, epochs=10)
prediction= model.predict(xt)
model_json = model.to_json()

with open("model.json", "w") as json_file:

    json_file.write(model_json)

# serialize weights to HDF5

model.save_weights("model.h5")

print("Saved model to disk")
prediction[4:10]
yt[4:10]
pred=[]

for i in prediction:

    r=[]

    for j in i:

        if j>0.5:

            a=1.0

        else:

            a=0.0

        r.append(a)

    pred.append(r)



        

    

    
pred=np.array(pred,dtype="float32")
len(pred)
len(y_test)
count=0

for i in range(len(pred)):

    if (pred[i]==y_test[i]).all():

        count+=1

    else:

        print(pred[i],"diff",y_test[i])

        

    
count/3000
def test_dataset():

    dataset=[]

    for images in os.listdir("/kaggle/input/DataSet/Test Images/Test Images"):

        img=cv2.imread(os.path.join("/kaggle/input/DataSet/Test Images/Test Images",images),0)

        img=cv2.resize(img,(200,200))

        dataset.append(img)

        

    return dataset

test_data=test_dataset()
test=np.array(test_data)

test=test/255
test.shape
test_pred=np.expand_dims(test,axis=4)
test_prediction= model.predict(test_pred)
test_prediction
os.getcwd()
os.chdir("/kaggle/working")
tpred=[]

for i in test_prediction:

    r=[]

    for j in i:

        if j>0.5:

            a=1.0

        else:

            a=0.0

        r.append(a)

    tpred.append(r)

tpred=np.array(tpred,dtype="float32")
keras.utils.to_categorical(0, 2)
tpred[:5]
X=[]

for i in tpred:

    if int(i[0])==1:

        X.append("Small")

    else:

        X.append("Large")

    

    
import pickle
pickle_out = open("result.pickle","wb")

pickle.dump(X, pickle_out)

pickle_out.close()
X
pickle_out = open("resulttest.pickle","wb")

pickle.dump(X, pickle_out)

pickle_out.close()