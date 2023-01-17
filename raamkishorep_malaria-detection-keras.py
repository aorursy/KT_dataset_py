#Importing Necessary Libraries.

from PIL import Image

import numpy as np

import os

import cv2

import keras

from keras.utils import np_utils

from keras.models import Sequential

from keras.layers import Conv2D,MaxPooling2D,Dense,Flatten,Dropout
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
print("Reading Images")
data=[]

labels=[]

Parasitized=os.listdir("../input/cell-images-for-detecting-malaria/cell_images/Parasitized/")

try:

    for a in Parasitized:

        if a.endswith(".png"):

            image=cv2.imread("../input/cell-images-for-detecting-malaria/cell_images/Parasitized/"+a)

            image_from_array = Image.fromarray(image, 'RGB')

            size_image = image_from_array.resize((50, 50))

            data.append(np.array(size_image))

            labels.append(0)

except:

    print("Error")





Uninfected=os.listdir("../input/cell-images-for-detecting-malaria/cell_images/Uninfected/")

try:

    for b in Uninfected:

        if b.endswith(".png"):

            image=cv2.imread("../input/cell-images-for-detecting-malaria/cell_images/Uninfected/"+b)

            image_from_array = Image.fromarray(image, 'RGB')

            size_image = image_from_array.resize((50, 50))

            data.append(np.array(size_image))

            labels.append(1)

except:

    print("Error")



    
Cells=np.array(data)

labels=np.array(labels)
np.save("Cells",Cells)

np.save("labels",labels)
Cells=np.load("Cells.npy")

labels=np.load("labels.npy")
s=np.arange(Cells.shape[0])

np.random.shuffle(s)

Cells=Cells[s]

labels=labels[s]
num_classes=len(np.unique(labels))

len_data=len(Cells)
(x_train,x_test)=Cells[(int)(0.1*len_data):],Cells[:(int)(0.1*len_data)]

x_train = x_train.astype('float32')/255 # As we are working on image data we are normalizing data by divinding 255.

x_test = x_test.astype('float32')/255

train_len=len(x_train)

test_len=len(x_test)
(y_train,y_test)=labels[(int)(0.1*len_data):],labels[:(int)(0.1*len_data)]
#Doing One hot encoding as classifier has multiple classes

y_train=keras.utils.to_categorical(y_train,num_classes)

y_test=keras.utils.to_categorical(y_test,num_classes)
#creating sequential model

model=Sequential()

model.add(Conv2D(filters=16,kernel_size=2,padding="same",activation="relu",input_shape=(50,50,3)))

model.add(MaxPooling2D(pool_size=2))

model.add(Conv2D(filters=32,kernel_size=2,padding="same",activation="relu"))

model.add(MaxPooling2D(pool_size=2))

model.add(Conv2D(filters=64,kernel_size=2,padding="same",activation="relu"))

model.add(MaxPooling2D(pool_size=2))

model.add(Dropout(0.2))

model.add(Flatten())

model.add(Dense(500,activation="relu"))

model.add(Dropout(0.2))

model.add(Dense(2,activation="sigmoid"))#2 represent output layer neurons 

model.summary()
# compile the model with loss as categorical_crossentropy and using adam optimizer you can test result by trying RMSProp as well as Momentum

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print("Training...........")
#Fit the model with min batch size as 50[can tune batch size to some factor of 2^power ] 

model.fit(x_train,y_train,batch_size=50,epochs=3,verbose=1)
accuracy = model.evaluate(x_test, y_test, verbose=1)

print('\n', 'Test_Accuracy:-', accuracy[1])
from keras.models import load_model

model.save('cells.h5')
from keras.models import load_model

import cv2

import numpy as np

#modell = load_model('/kaggle/input/modell/cells.h5')

pic = cv2.imread('/kaggle/input/testing/unaffected.png')

pic = cv2.cvtColor(pic, cv2.COLOR_BGR2RGB)

pic = cv2.resize(pic,(50,50))

pic = pic/255

pic = np.reshape(pic,[1,50,50,3])

y_prob = model.predict(pic) 

y_classes = y_prob.argmax(axis=-1)

if y_classes == 0:

    print("Affected")

if y_classes == 1:

    print("Unaffected")
