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
import numpy as np;
import pandas as pd;
import seaborn as sns;
import os;
import matplotlib.pyplot as plt;
import matplotlib.image as mpimg;
#%matplotlib inline

from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau
from sklearn.metrics import confusion_matrix;
import itertools;
from sklearn.model_selection import train_test_split;

df_train= pd.read_csv("../input/digit-recognizer/train.csv");

df_test= pd.read_csv("../input/digit-recognizer/test.csv");




Y_train=df_train["label"];
X_train=df_train.drop(labels=["label"],axis=1);


df_test=df_test/255.0;
X_train=X_train/255.0;


df_test=df_test.values.reshape(-1,28,28,1);
X_train=X_train.values.reshape(-1,28,28,1);

Y_train=to_categorical(Y_train,num_classes=10);
#X_norm_onehot=to_categorical(X_train,num_classes=41999);


print(Y_train);


np.random.seed(2);
X_train,X_val,Y_train,Y_val= train_test_split(X_train,Y_train,test_size=0.2,random_state=2);



model=Sequential();


model.add(Conv2D(filters=32,kernel_size=(5,5),padding='Same',activation='relu',input_shape=(28,28,1)));
model.add(Conv2D(filters=32,kernel_size=(5,5),padding='Same',activation='relu'));
model.add(MaxPool2D(pool_size=(2,2)));
model.add(Dropout(0.25));

model.add(Conv2D(filters=64,kernel_size=(3,3),padding='Same',activation='relu'));
model.add(Conv2D(filters=64,kernel_size=(3,3),padding='Same',activation='relu'));
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)));
model.add(Dropout(0.25));
model.add(Flatten());
model.add(Dense(256,activation='relu'));
model.add(Dropout(0.5));
model.add(Dense(10,activation='softmax'));


print(X_val,Y_val);
optimizer=RMSprop(lr=0.001,rho=0.9,epsilon=1e-08,decay=0.0);
model.compile(optimizer=optimizer,loss='categorical_crossentropy',metrics=["accuracy"]);



lr_reduction=ReduceLROnPlateau(monitor='val_acc',patience=3,verbose=1,factor=0.5,min_lr=0.00001);
epochs=50;
batch_size=86;

history = model.fit(X_train, Y_train, batch_size = batch_size, epochs = epochs,validation_data = (X_val, Y_val), verbose = 2)
results = model.predict(df_test)

# select the indix with the maximum probability
results = np.argmax(results,axis = 1)

results = pd.Series(results,name="Label")
print(results)
submission=pd.concat([pd.Series(range(1,28001),name="ImageId"),results],axis=1)
submission.to_csv("cnn_dataset_mnist",index=False);