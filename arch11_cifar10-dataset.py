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
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
from tensorflow.keras.datasets import cifar10
(X_train,y_train),(X_test,y_test) = cifar10.load_data()
print(X_train.shape , X_test.shape)
plt.imshow(X_train[12])
X_train = X_train/255
X_test = X_test/255
y_train
from tensorflow.keras.utils import to_categorical
y_cat_train = to_categorical(y_train,10)
y_cat_test = to_categorical(y_test,10)
#categorical label names

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D,Flatten,Dropout, AveragePooling2D
32*32*3
model = Sequential()
model.add(Conv2D(filters= 32, kernel_size=(3,3) , strides=(1,1) ,padding = "same",input_shape=(32,32,3),activation="relu"))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.2))

model.add(Conv2D(filters= 64, kernel_size=(3,3) , strides=(1,1) ,padding = "same",activation="relu"))
model.add(Dropout(0.2))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Conv2D(filters= 128, kernel_size=(3,3) , strides=(1,1) ,padding = "same",activation="relu"))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.2))

model.add(Conv2D(filters= 256, kernel_size=(3,3) , strides=(1,1) ,padding = "same",activation="relu"))
model.add(Dropout(0.2))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Conv2D(filters= 512, kernel_size=(3,3) , strides=(1,1) ,padding = "same",activation="relu"))
model.add(Dropout(0.2))
model.add(AveragePooling2D(pool_size=(2,2)))

model.add(Flatten())

model.add(Dense(512,activation="relu"))

model.add(Dense(256,activation="relu"))
#model.add(Dropout(0.2))
model.add(Dense(128,activation="relu"))
#model.add(Dropout(0.2))
model.add(Dense(64,activation="relu"))

model.add(Dense(10,activation="softmax"))

model.compile(loss="categorical_crossentropy",optimizer="adam",metrics=["accuracy"])
model.summary()
from tensorflow.keras.callbacks import EarlyStopping
early_stop = EarlyStopping(monitor="val_loss",patience=3,restore_best_weights=True)
model.fit(X_train,y_cat_train,validation_data=(X_test,y_cat_test),epochs=50,callbacks=[early_stop])
metrics = pd.DataFrame(model.history.history)
metrics[["accuracy","val_accuracy"]].plot()
metrics[["loss","val_loss"]].plot()
model.evaluate(X_test,y_cat_test,verbose=0)
from sklearn.metrics import classification_report, confusion_matrix
pred = model.predict_classes(X_test)
print(classification_report(y_test,pred))
print(confusion_matrix(y_test,pred))
plt.figure(figsize=(10,6))
sns.heatmap(confusion_matrix(y_test,pred),annot=True)
plt.imshow(X_test[0])
model.predict_classes(X_test[0].reshape(1,32,32,3))    #3 is cat
plt.imshow(X_test[16])   #16 is dog
model.predict_classes(X_test[16].reshape(1,32,32,3)) 
