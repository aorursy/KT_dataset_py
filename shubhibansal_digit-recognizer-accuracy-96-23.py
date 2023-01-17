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
import os
from os import listdir
listdir('/kaggle/input')
listdir('/kaggle/input/digit-recognizer')
import pandas as pd
data=pd.read_csv("/kaggle/input/digit-recognizer/train.csv")
data
data.head()
y=data.iloc[:,0:1]
data.shape
y
data=data.drop("label",axis=1)
y.value_counts
x=[]
import numpy as np
for i in range(42000):
    y1=data.iloc[i:i+1,:]
    y1=np.array(y1)
    y1=y1.reshape(28,28)
    y1=y1.reshape(28,28,1)
    x.append(y1)
    
len(x)
x=np.array(x)
x.shape
y.shape
y=y.values
y.shape
from keras.utils import to_categorical
trainy=to_categorical(y)
trainy.shape
x.max()
trainx=x/55
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(trainx, trainy, test_size = 1/3, random_state = 0)
from keras import Model
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.layers import Dropout
model=Sequential()
def ima():
  model=Sequential()
  model.add(Conv2D(10,(3,3),padding="same",activation="relu",input_shape=(28, 28, 1)))
  model.add(MaxPooling2D((2,2),strides=(2,2)))
  model.add(Flatten())
  model.add(Dense(16, activation="relu", kernel_initializer="he_uniform"))
  model.add(Dense(10, activation="softmax"))
  opt = Adam(lr=0.01)
  model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["accuracy"])
  return model 
k1=ima()
history=k1.fit(x_train, y_train,validation_data=(x_test,y_test), epochs=4, batch_size=32,verbose=1)
from matplotlib import pyplot
pyplot.title("Classification Accuracy")
pyplot.plot(history.history["accuracy"], color="blue", label="train")
pyplot.plot(history.history["val_accuracy"], color="orange", label="test")
