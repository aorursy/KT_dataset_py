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

import numpy as np

import tensorflow.keras
import matplotlib.pyplot as plt

%matplotlib inline
from tensorflow.keras.datasets import mnist

(X_train,y_train), (X_test, y_test) = mnist.load_data()
X_train.shape
X_test.shape
plt.imshow(X_train[15],cmap="Greys")
y_train.shape
y_test.shape
X_train[0]
y_train
from tensorflow.keras.utils import to_categorical
y_cat_train=to_categorical(y_train,num_classes=10)

y_cat_test = to_categorical(y_test,num_classes=10)
X_train = X_train/255

X_test = X_test/255
X_train.shape
X_train = X_train.reshape(60000,28,28,1)

X_test = X_test.reshape(10000,28,28,1)
from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense,Conv2D,MaxPool2D, Flatten
model = Sequential()



model.add(Conv2D(filters=32, kernel_size=(4,4),strides = (1,1),padding="valid",input_shape=(28,28,1),activation="relu"))

model.add(MaxPool2D(pool_size=(2,2)))



model.add(Flatten())

model.add(Dense(128,activation="relu"))

model.add(Dense(32,activation="relu"))

model.add(Dense(10,activation="softmax"))



model.compile(optimizer='adam',loss="categorical_crossentropy",metrics=["accuracy"])

from tensorflow.keras.callbacks import EarlyStopping

early_stop = EarlyStopping(monitor="val_loss",patience=1)
import tensorflow as tf

tf.config.list_physical_devices(

    device_type=None

)
with tf.device("/GPU:0"):

    model.fit(X_train,y_cat_train,epochs=25,validation_data=(X_test,y_cat_test),callbacks=[early_stop])

    
losses = pd.DataFrame(model.history.history)

losses[["accuracy","val_accuracy"]].plot()
losses[["loss","val_loss"]].plot()
pred = model.predict_classes(X_test)
model.evaluate(X_test,y_cat_test,verbose=0)
from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_test,pred))
print(confusion_matrix(y_test,pred))
import seaborn as sns

sns.heatmap(confusion_matrix(y_test,pred),annot=True)
my_number = X_test[0]

plt.imshow(my_number.reshape(28,28))
model.predict_classes(my_number.reshape(1,28,28,1))