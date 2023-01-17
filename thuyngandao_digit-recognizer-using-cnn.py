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
train = pd.read_csv("../input/train.csv")

X_test= pd.read_csv("../input/test.csv")

Y_train = train["label"]

X_train = train.drop(labels = ["label"],axis = 1) 

del train
X_train = X_train / 255.0

X_test = X_test / 255.0
img_rows = img_cols = 28

X_train = X_train.values.reshape(X_train.shape[0],img_rows,img_cols, 1)

X_test = X_test.values.reshape(X_test.shape[0],img_rows,img_cols, 1)
from tensorflow import keras

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPool2D
model = Sequential()

model.add(Conv2D(16, kernel_size=(3, 3), activation='relu', input_shape=(img_rows, img_cols, 1)))

model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))

model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))

model.add(MaxPool2D(pool_size=(2,2)))

model.add(Dropout(0.25))



model.add(Flatten())

model.add(Dense(256, activation='relu'))

model.add(Dropout(0.2))

model.add(Dense(10, activation='softmax'))

    

model.compile(optimizer='adam',

      loss='sparse_categorical_crossentropy',

      metrics=['accuracy'])
model.fit(X_train, Y_train, epochs=30, validation_split=0.2, verbose=2)
preds = model.predict_classes(X_test)
np.savetxt('submission.csv', 

           np.c_[range(1,len(X_test)+1),preds], 

           delimiter=',', 

           header = 'ImageId,Label', 

           comments = '', 

           fmt='%d')