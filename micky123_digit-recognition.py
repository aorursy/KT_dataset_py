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
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split

from keras.utils.np_utils import to_categorical

from keras.models import Sequential

from keras.layers import Conv2D, Dense, MaxPool2D, Flatten, Dropout

from sklearn.metrics import confusion_matrix, precision_score, recall_score

from keras.optimizers import RMSprop

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ReduceLROnPlateau
#Splitting data into train and test.
train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")
train.head()
y_train = train["label"]

x_train = train.drop(["label"], axis = 1)

y_train.value_counts()
sns.countplot(train["label"], palette = "Greens")
x_train.isnull().sum()
y_train.isnull().sum()
test.isnull().sum()
x_train = x_train/255.0

test = test/255.0

random_seed = 2
x_train = x_train.values.reshape(-1,28,28,1)

test = test.values.reshape(-1,28,28,1)
y_train = to_categorical(y_train, num_classes = 10)
g = plt.imshow(x_train[0][:,:,0])
model = Sequential()

model.add(Conv2D(filters = 32, kernel_size = (5,5), padding = "Same", activation = "relu", input_shape = (28,28,1) ))

model.add(Conv2D(filters = 32, kernel_size = (5,5), padding = "Same", activation = "relu"))

model.add(Conv2D(filters = 32, kernel_size = (5,5), padding = "Same", activation = "relu"))

model.add(MaxPool2D(pool_size = ((2,2))))

model.add(Dropout(0.25))

model.add(Conv2D(filters = 32, kernel_size = (5,5), padding = "Same", activation = "relu"))

model.add(Conv2D(filters = 32, kernel_size = (5,5), padding = "Same", activation = "relu"))

model.add(MaxPool2D(pool_size = (2,2), strides = (2,2)))

model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(256, activation = "relu"))

model.add(Dropout(0.5))

model.add(Dense(10, activation = "softmax"))
optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size = 0.1,

                                                  random_state=random_seed)
epochs = 30

batch_size = 86

model.compile(optimizer = optimizer , 

              loss = "categorical_crossentropy", 

              metrics=["accuracy"])

learning_rate_reduction = ReduceLROnPlateau(

    monitor = "val_acc", patience = 3, verbose = 1, 

    factor = 0.5, min_lr = 0.00001)
history = model.fit(x_train, y_train, batch_size = batch_size, epochs = epochs,  validation_data = (x_val, y_val), verbose = 2)
y_pred = model.predict(x_val)

y_pred_classes = np.argmax(y_pred,axis = 1) 

y_true = np.argmax(y_val,axis = 1) 
confusion_mtx = confusion_matrix(y_true, y_pred_classes) 

results = model.predict(test)

results = np.argmax(results,axis = 1)

results = pd.Series(results,name="Label")
submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)

submission.to_csv("Digit_recognition_submission.csv",index=False)