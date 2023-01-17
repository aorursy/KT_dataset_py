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



# Any results you write to the current directory are saved as output.
import matplotlib.image as mpimg

import matplotlib.pyplot as plt

import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.optimizers import RMSprop

from keras.utils import np_utils

import random
train=pd.read_csv("../input/digit-recognizer/train.csv")

test=pd.read_csv("../input/digit-recognizer/test.csv")

submission=pd.read_csv("../input/digit-recognizer/sample_submission.csv")
print(train.head(1))

y=train['label']

X=train.drop(['label'],axis=1)
# visualizing some random digits

X=np.array(X)

X=X.reshape(-1,28,28)

print(X.shape)

fig,ax=plt.subplots(3,3)

for i in range(3):

    for j in range(3):

        ax[i,j].imshow(random.choice(X))
X=X.reshape(-1,28,28,1)  # --> reshaping the pixels list  into a 2-D pixel grid

y=np_utils.to_categorical(y,10) # --> hot encoding
model=tf.keras.models.Sequential([ tf.keras.layers.Conv2D(16,(3,3),activation='relu',input_shape=(28,28,1)),

                                   tf.keras.layers.MaxPooling2D((2,2)),

                                   tf.keras.layers.Conv2D(32,(3,3),activation='relu'),

                                   tf.keras.layers.MaxPooling2D((2,2)),

                                   tf.keras.layers.Conv2D(64,(3,3),activation='relu'),

                                   tf.keras.layers.MaxPooling2D((2,2)),

                                   tf.keras.layers.Flatten(),

                                   tf.keras.layers.Dense(128,activation='relu'),

                                   tf.keras.layers.Dense(10,activation='softmax')

    

    

    

])
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.1,random_state=0)
print("X_train Shape:"+ str(X_train.shape))

print("y_train Shape:"+ str(y_train.shape))

print("X_test Shape:"+ str(X_test.shape))

print("y_test Shape:"+ str(y_test.shape))

print("X Shape:"+ str(X.shape))

print("y Shape:"+ str(y.shape))

# Normalizing the data

X_train=X_train/255.0

X_test=X_test/255.0
#compiling the model

model.compile(loss='categorical_crossentropy',optimizer='RMSprop',metrics=['accuracy'])
history=model.fit(X_train,y_train,epochs=10,validation_data=(X_test,y_test))
# saving model 

model.save("MNIST-14-09-2019-02.h5")
plt.plot(history.history['acc'])

plt.plot(history.history['val_acc'])

plt.legend(['Training','Validation'])

plt.title("Model accuracy ")

plt.xlabel("Epochs")
test=np.array(test)

test=test.reshape(-1,28,28,1)

y_pred=model.predict(test)
y_pred=model.predict(test)

y_pred_df=pd.DataFrame(y_pred.T)

y_pred_df=y_pred_df.idxmax(axis=0)

test=test.reshape(-1,28,28)

a=np.random.randint(1,30)

plt.imshow(test[a])

plt.title("Prediction:"+str(y_pred_df[a]))



        

        
submission=pd.concat([submission,y_pred_df],axis=1)

submission.drop(['Label'],axis=1,inplace=True)

submission.columns=['ImageId','Label']

submission.to_csv("MNIST-14-09-19-02.csv",index=False)
model.evaluate(X_test,y_test) 