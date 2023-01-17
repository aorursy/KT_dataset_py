

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)





import os

print(os.listdir("../input/leapgestrecog/leapGestRecog"))



# Any results you write to the current directory are saved as output.
import os

import numpy as np

from PIL import Image

import matplotlib.image as mpimg 

import numpy as np

import matplotlib.pyplot as plt

import matplotlib.cm as cm

import IPython.display

path='../input/leapgestrecog/leapGestRecog'

folders=os.listdir(path)

folders=set(folders)



import codecs

import json





different_classes=os.listdir(path+'/'+'00')

different_classes=set(different_classes)









print("The different classes that exist in this dataset are:")

print(different_classes,sep='\n')









classes=dict()

names_of_classes=dict()

for i in different_classes:

    classes[int(i.split('_')[0])]='_'.join(i.split('_')[1:])

    names_of_classes['_'.join(i.split('_')[1:])]=int(i.split('_')[0])









print(classes)









x=[]

z=[]

y=[]#converting the image to black and white

threshold=200

import cv2





for i in folders:

    print('***',i,'***')

    subject=path+'/'+i

    subdir=os.listdir(subject)

    subdir=set(subdir)

    for j in subdir:

        print(j)

        images=os.listdir(subject+'/'+j)

        for k in images:

            results=dict()

            results['y']=j.split('_')[0]

            img = cv2.imread(subject+'/'+j+'/'+k,0)

            img=cv2.resize(img,(int(160),int(60)))

            

            ret, imgf = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

            imgD=np.asarray(img,dtype=np.float64)

            z.append(imgD)

            imgf=np.asarray(imgf,dtype=np.float64)

            x.append(imgf)

            y.append(int(j.split('_')[0]))

            results['x']=imgf



print(list(set(y)))

        

# import pandas as pd 

# df=pd.DataFrame({'x':x,'y':y})

# df.to_csv('results.csv',index=False)

#sample black and white image from each class

l = []

list_names = []

for i in range(10):

    l.append(0)

for i in range(len(x)):

    if(l[y[i] - 1] == 0):

        l[y[i] - 1] = i

        if(len(np.unique(l)) == 10):

            break

for i in range(len(l)):

    %matplotlib inline

    print("Class Label: " + str(i + 1))

    plt.imshow(np.asarray(z[l[i]]), cmap  =cm.gray)

    plt.show()

    plt.imshow(np.asarray(x[l[i]]), cmap = cm.gray)     

    plt.show()

    

x=np.array(x)

y=np.array(y)

y = y.reshape(len(x), 1)

print(x.shape)

print(y.shape)

print(max(y),min(y))
np.unique(y)
import keras

from keras.utils import to_categorical

def one_hot_encoded(y):

    p=list(np.unique(y))

    dictionary=dict()

    final_result=[]

    for i in range(len(p)):

        dictionary[p[i]]=i

    for i in y:

        actual=[0 for j in range(len(p))]

        actual[dictionary[i[0]]]=1

        final_result.append(actual)

    return np.array(final_result)

        

        

    

y_data = one_hot_encoded(y)

print(y_data[0])

print(type(y_data))
x_data = x.reshape((len(x), 60, 160, 1))



x_data/=255

print(x_data.shape)
from sklearn.model_selection import train_test_split

x_train,x_further,y_train,y_further = train_test_split(x_data,y_data,test_size = 0.2)

x_validate,x_test,y_validate,y_test = train_test_split(x_further,y_further,test_size = 0.5)
from keras import layers

from keras import models
model=models.Sequential()

model.add(layers.Conv2D(32, (5, 5), strides=(2, 2), activation='relu', input_shape=(60, 160,1))) 

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(64, (3, 3), activation='relu')) 

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Flatten())

model.add(layers.Dense(128, activation='relu'))

model.add(layers.Dense(10, activation='softmax'))
model.compile(optimizer='rmsprop',

              loss='categorical_crossentropy',

              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=10, batch_size=64, verbose=1, validation_data=(x_validate, y_validate))
model.save('cnn_results.h5')
[loss, acc] = model.evaluate(x_test,y_test,verbose=1)

print("Accuracy:" + str(acc))