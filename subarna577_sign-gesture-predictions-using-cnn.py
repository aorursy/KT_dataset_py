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
import pandas as pd

import matplotlib.pyplot as plt

import seaborn as ssn

import numpy as np

import keras

from keras.layers import Dense, Conv2D, Flatten, BatchNormalization, Dropout, MaxPool2D

from keras.models import Sequential



from sklearn.metrics import confusion_matrix, accuracy_score

from sklearn.model_selection import train_test_split



from pandas_profiling import ProfileReport

import time

from sklearn.preprocessing import LabelBinarizer

from keras.preprocessing.image import ImageDataGenerator





base={0:'A',1:'B',2:'C',3:'D',4:'E',5:'F',6:'G',7:'H',8:'I',9:'J',10:'K',11:'L',12:'M',13:'N',

      14:'O',15:'P',16:'Q',17:'R',18:'S',19:'T',20:'U',21:'V',22:'W',23:'X',24:'Y',25:'Z'}





# Using tensorflow as backend
train_data= pd.read_csv('../input/sign-language-mnist/sign_mnist_train.csv')

test_data= pd.read_csv('../input/sign-language-mnist/sign_mnist_test.csv')

train_data.head()
test_data.head()
train_labels=np.array(train_data['label'])

del train_data['label']



x_train= train_data.values

x_train= x_train.reshape(-1,28,28,1)





f, ax = plt.subplots(2,4)



index=0



for i in range(2):

    for j in range(4):

        ax[i,j].imshow(x_train[index].reshape(28,28),cmap='gray')

        index+=1



        

#Images after normalising the inputs





x_train=x_train/255.



w, mx= plt.subplots(4,4)



k=0

for i in range(4):

    for j in range(4):

        mx[i,j].imshow(x_train[k].reshape(28,28),cmap='binary')

        mx[i,j].set_title(base[train_labels[k]])

        

        k+=1

        

    plt.tight_layout()
print("The total number of labels are {}".format(len(set(train_labels))))
train_labels_char=[]

for i in range(len(train_labels)):

    train_labels_char.append(base[(train_labels[i])])



train_labels_char=sorted(train_labels_char)



ssn.countplot(train_labels_char)
test_labels=test_data['label']



y=test_data['label']



del test_data['label']

test_data=test_data.values

test_data=test_data.reshape(-1,28,28,1)
#Checking the uniformity of testing labels



ssn.countplot(test_labels)
import random

x_test=test_data

x_test=x_test/255.

x_test=x_test.reshape(-1,28,28,1)

print("The number of variables in x_test is {}".format(len(x_test)))

plt.imshow(x_test[random.choice(range(1,len(x_test)))].reshape(28,28),cmap='binary')
#To overcome overfitting problem, I am trying to increase the training data by using ImageDataGenerator from the existing training data.



dgen= ImageDataGenerator(featurewise_center=False,samplewise_center=False,featurewise_std_normalization=False,

                        samplewise_std_normalization=False,zca_whitening=False, rotation_range=10, zoom_range=0.1,

                        width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=False, vertical_flip=False)

dgen.fit(x_train)


model=Sequential()





#Initial Conv2D layer with 64 filters with filter sizes of 3X3

model.add(Conv2D(64,(3,3),strides=2,padding='same',activation='elu',input_shape=(28,28,1)))

#Although RELU is common, I am experimenting with ELU. ELU is similar to RELU except negative inputs.



model.add(MaxPool2D((2,2),strides=2,padding='same'))



#Second Conv2D layer with 50 filters and 3X3 filter size



model.add(Conv2D(50,(3,3),strides=1,padding='same',activation='elu'))



#Third Conv2D layer with 128 filters and 3X3 filter size



model.add(Conv2D(128,(3,3),strides=1,padding='same',activation='elu'))



model.add(Flatten())



#Output Layer

model.add(Dense(units=24,activation='softmax'))



model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])



model.summary()
#Using TPU v3-8

# x_train, train_labels, x_test, test_labels

l_binarizer=LabelBinarizer()

train_labels= l_binarizer.fit_transform(train_labels)

test_labels= l_binarizer.fit_transform(test_labels)



start_time=time.time()

history=model.fit(dgen.flow(x_train,train_labels,batch_size=50),epochs=20, validation_data=(x_test,test_labels))

end_time=time.time()
mins= int(round(end_time-start_time,2)//60)

seconds= round(round(end_time-start_time,2)%60)

print("The time taken to build the model is {} minutes {} seconds".format(mins,seconds))
print("The accuracy of the model is ",model.evaluate(x_test,test_labels)[1]*100,'%')
training_accuracy=history.history['accuracy']

training_loss= history.history['loss']

validation_accuracy=history.history['val_accuracy']

validation_loss= history.history['val_loss']

epochs=list(range(1,21))

#Accuracy Part



m, ax= plt.subplots(1,2)

ax[0].plot(epochs,training_accuracy,'yo--',label='Train Accuracy')

ax[0].plot(epochs, validation_accuracy,'bo--',label='Validation Accuracy')

ax[0].set_title('Accuracy')

ax[0].legend()





#Loss Part



ax[1].plot(epochs, training_loss, 'yo-',label='Training Loss')

ax[1].plot(epochs, validation_loss,'bo-',label='Validation Loss')

ax[1].set_title("Loss")

ax[1].legend()
pd= model.predict_classes(x_test)

plt.figure(figsize=(16,16))

cm= confusion_matrix(y,pd)

ssn.heatmap(cm, fmt="d",cmap='PuRd',annot=True,linewidths=1,square=True)
total=0



f, mx= plt.subplots(4,2)

k=300

for i in range(4):

    for j in range(2):

        mx[i,j].imshow(x_test[k].reshape(28,28),cmap='pink')

        mx[i,j].set_title("Predicted: {} Actual: {}".format(pd[k],y[k]))

        if pd[k]==y[k]:

            total+=1

        k+=1

        

    plt.tight_layout()

print("Total correctly classified results from random k is {} out of 8".format(total))