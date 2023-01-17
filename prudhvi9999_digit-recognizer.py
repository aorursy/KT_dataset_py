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
import tensorflow as tf
data=pd.read_csv('/kaggle/input/digit-recognizer/train.csv')
train_y=data['label']
train_X=data.drop(labels=['label'],axis=1)
test=pd.read_csv('/kaggle/input/digit-recognizer/test.csv')
test.head()
print(train_X.isnull())

print(train_y.isnull())
train_X,test=train_X/255.0,test/255.0
train_X = train_X.values.reshape(-1,28,28,1)

test = test.values.reshape(-1,28,28,1)
train_X.shape
train_y = tf.keras.utils.to_categorical(train_y, num_classes = 10)
from sklearn.model_selection import train_test_split
X_train,X_val,y_train,y_val=train_test_split(train_X,train_y,test_size=0.2)
import matplotlib.pyplot as plt

plt.imshow(X_train[0][:,:,0])

print(np.argmax(y_train[0]))
from tensorflow.keras import Sequential

from tensorflow.keras.layers import Flatten,Dense
earlystopping=tf.keras.callbacks.EarlyStopping(monitor='val_acc', patience=10)
def get_model():

    model=Sequential([

        Flatten(input_shape=(28,28,1)),

        Dense(512,activation='relu'),

        Dense(256,activation='relu'),

        Dense(10,activation='softmax')

    ])

    return model
model=get_model()
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
model.fit(X_train,y_train,epochs=10,batch_size=8,callbacks=[earlystopping],validation_data=(X_val,y_val))
import matplotlib.pyplot as plt

loss=model.history.history.get('loss')

acc=model.history.history.get('accuracy')

plt.plot(loss)

plt.plot(acc)

plt.xlabel('Loss')

plt.ylabel('Accuracy')

plt.title('Model Loss and Accuracy')
results=model.predict(test)

results=np.argmax(results,axis=1)

results = pd.Series(results,name="Label")
res=tf.keras.utils.to_categorical(results)

model.evaluate(test,res)
submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)



submission.to_csv("mnist.csv",index=False)