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
train=pd.read_csv("/kaggle/input/Kannada-MNIST/train.csv")

test=pd.read_csv("/kaggle/input/Kannada-MNIST/test.csv")
X=train.iloc[:,1:].values

y=train.iloc[:,1].values
#reshape data

X = X.reshape(X.shape[0], 28, 28)/255.0

X = X.reshape(X.shape[0], 28,28,1)
test=test.iloc[:,1:].values
test = test.reshape(test.shape[0], 28, 28)/255.0

test = test.reshape(test.shape[0], 28,28,1)
#import keras libraries

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Convolution2D

from tensorflow.keras.layers import MaxPooling2D

from tensorflow.keras.layers import Flatten

from tensorflow.keras.layers import Dense,Dropout
#initialize

classifier=Sequential()
#layer1

classifier.add(Convolution2D(28,3,3,input_shape=(28,28,1),activation='relu'))

classifier.add(MaxPooling2D(pool_size=(2,2)))

classifier.add(Dropout(0.25))
#layer2

classifier.add(Convolution2D(32,padding='same',kernel_size=3,activation='relu'))

classifier.add(MaxPooling2D(pool_size=(2,2)))
#layer3

classifier.add(Convolution2D(64,padding='same',kernel_size=3,activation='relu'))

classifier.add(MaxPooling2D(pool_size=(2,2)))
#flattening

classifier.add(Flatten())
classifier.add(Dense(units=128,activation='relu'))

classifier.add(Dropout(0.5))

classifier.add(Dense(units=10,activation='softmax'))
#compile

classifier.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
#fit model

classifier.fit(X, y, batch_size = 12, epochs = 20)
#predict

y_pred = classifier.predict(test)
import numpy as np

results = np.argmax(y_pred,axis = 1)

data_out = pd.DataFrame({'id': range(len(test)), 'label': results})

data_out.to_csv('try.csv', index = None)