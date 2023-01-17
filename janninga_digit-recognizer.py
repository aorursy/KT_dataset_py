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
train_data = pd.read_csv("/kaggle/input/digit-recognizer/train.csv")

test_data = pd.read_csv("/kaggle/input/digit-recognizer/test.csv")



train_data.head()
y = train_data["label"]

y = y.values[:]



num_classes = 10

img_rows = 28

img_cols = 28

num_images = len(y)



X_array = train_data.values[:,1:]

X = X_array.reshape(-1, img_rows, img_cols, 1)

X = X/225



test_array = test_data.values[:,:]

test = test_array.reshape(-1, img_rows, img_cols, 1)

test = test/225



#X_train,X_val,y_train,y_val=train_test_split(X,y,test_size=0.1)

from sklearn.model_selection import train_test_split

from tensorflow.python import keras

from tensorflow.python.keras.models import Sequential

from tensorflow.python.keras.layers import Dense, Flatten, Conv2D, Dropout



model = Sequential()

model.add(Conv2D(40, kernel_size=(6, 6),

                 activation='relu',

                 input_shape=(img_rows, img_cols, 1)))

model.add(Conv2D(40, kernel_size=(6, 6), activation='relu'))

model.add(Flatten())

model.add(Dense(200, activation='relu'))

model.add(Dense(num_classes, activation='softmax'))



model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['accuracy'])



model.fit(X, y, batch_size=128, epochs=20, validation_split = 0.2)
test_labels = model.predict(test)

test_labels[1]
results = np.argmax(test_labels,axis = 1)

results = pd.Series(results,name="Label")

submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)



submission.to_csv("submission_final.csv",index=False)