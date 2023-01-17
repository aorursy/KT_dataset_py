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
df=pd.read_csv('../input/train.csv')

df.head()
train=df.drop(['label'],axis=1)
train.head()
labels=df['label']
labels.head()
X,Y=train.values,labels.values
X=X.reshape(X.shape[0],28,28)

X=X/255.
import matplotlib.pyplot as plt

fig=plt.gcf()

fig.set_size_inches(9,9)

for i,img in enumerate(X):

    if i+1>3*3:break

    plt.subplot(3,3,i+1)

    plt.imshow(img)

plt.show()    
import tensorflow as tf
model=tf.keras.Sequential([

    tf.keras.layers.Flatten(input_shape=(28,28)),

    tf.keras.layers.Dense(64,activation='relu'),

    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Dense(128,activation='relu'),

    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Dense(128,activation='relu'),

    tf.keras.layers.Dropout(0.2),

    

    tf.keras.layers.Dense(10,activation='softmax'),

    

])

model.compile(metrics=['acc'],optimizer='adam',loss='sparse_categorical_crossentropy')
model.summary()
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=.1)
history=model.fit(X_train,Y_train,validation_data=(X_test,Y_test),epochs=100)
#-----------------------------------------------------------

# Retrieve a list of list results on training and test data

# sets for each training epoch

#-----------------------------------------------------------

acc      = history.history[     'acc' ]

val_acc  = history.history[ 'val_acc' ]

loss     = history.history[    'loss' ]

val_loss = history.history['val_loss' ]



epochs   = range(len(acc)) # Get number of epochs



#------------------------------------------------

# Plot training and validation accuracy per epoch

#------------------------------------------------

plt.plot  ( epochs,     acc )

plt.plot  ( epochs, val_acc )

plt.title ('Training and validation accuracy')

plt.figure()



#------------------------------------------------

# Plot training and validation loss per epoch

#------------------------------------------------

plt.plot  ( epochs,     loss )

plt.plot  ( epochs, val_loss )

plt.title ('Training and validation loss'   )
df_test=pd.read_csv('../input/test.csv')
df_test.head()
X_testing=df_test.values

X_testing=X_testing.reshape(X_testing.shape[0],28,28)

X_testing=X_testing/255.
pred=model.predict(X_testing)
pred=pred.argmax(axis=1)
pred[0:19]
submit=pd.read_csv('../input/sample_submission.csv')

submit["Label"]=pred

submit.to_csv('prediction.csv',index=False)