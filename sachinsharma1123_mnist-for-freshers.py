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
train=pd.read_csv('/kaggle/input/digit-recognizer/train.csv')
train.head(5)
test=pd.read_csv('/kaggle/input/digit-recognizer/test.csv')
y=train['label'].values
x=train.drop(['label'],axis=1).values
import matplotlib.pyplot as plt

plt.figure(figsize=(12,10))

for i in range(1,5):

    plt.subplot(10,4,i+1)

    plt.imshow(x[i].reshape((28,28)),interpolation='nearest')

    plt.xlabel(y[i])

plt.show()
import seaborn as sns

sns.countplot(y)
x=x/255

test=test/255
print('training data shape:',x.shape)

print('test data shape:',test.shape)

print('y values shape:',y.shape)
x=np.reshape(x,(-1,28,28,1))

test=np.reshape(test.values,(-1,28,28,1))
from tensorflow.keras.utils import to_categorical

y=to_categorical(y)
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=0,test_size=0.2)
import tensorflow as tf

from tensorflow.keras.layers import Dense

from tensorflow.keras.layers import Dropout

from tensorflow.keras.models import Sequential

from tensorflow.keras import initializers

from tensorflow.keras import regularizers

from tensorflow.keras.layers import BatchNormalization

from keras.layers import  Flatten, Conv2D, MaxPool2D

model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',kernel_initializer='he_normal',input_shape=(28,28,1)))

model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',kernel_initializer='he_normal'))

model.add(MaxPool2D((2, 2)))

model.add(Dropout(0.20))

model.add(Conv2D(64, (3, 3), activation='relu',padding='same',kernel_initializer='he_normal'))

model.add(Conv2D(64, (3, 3), activation='relu',padding='same',kernel_initializer='he_normal'))

model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Dropout(0.25))

model.add(Conv2D(128, (3, 3), activation='relu',padding='same',kernel_initializer='he_normal'))

model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(128, activation='relu'))

model.add(BatchNormalization())

model.add(Dropout(0.25))

model.add(Dense(10, activation='softmax'))

optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001)

acc=tf.keras.metrics.SparseCategoricalAccuracy()

model.compile(optimizer=optimizer,loss='categorical_crossentropy',metrics=['acc'])
model.summary()
history=model.fit(x_train,y_train,epochs=10,batch_size=64)
loss,accuracy=model.evaluate(x_test,y_test)
print('loss:',loss)

print('accuracy:',accuracy)
history.history.keys()
df=pd.DataFrame(history.history)
df.plot(y='loss',title='loss vs epoch',legend=False)
df.plot(y='acc',title='acc vs epcoh',legend=False)
predicted_classes = model.predict_classes(test)

submissions=pd.DataFrame({"ImageId": list(range(1,len(predicted_classes)+1)),

                         "Label": predicted_classes})
submissions.to_csv("asd.csv", index=False, header=True)