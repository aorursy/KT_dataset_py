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
train_data=pd.read_csv('/kaggle/input/digit-recognizer/train.csv')

test_data=pd.read_csv('/kaggle/input/digit-recognizer/test.csv')
train_data.head()
# No data for train_data is missing

Col_with_missing = [col for col in train_data.columns if train_data[col].isnull().any()]

print(Col_with_missing)
# No data for test_data is missing

Col_with_missing = [col for col in test_data.columns if test_data[col].isnull().any()]

print(Col_with_missing)
test_data.head()
Y_train = train_data["label"]



# Drop 'label' column

X_train = train_data.drop(["label"],axis = 1) 
Y_train.value_counts()
import seaborn as sns

%matplotlib inline

sns.countplot(Y_train)
img_row=28

img_col=28

def data_prep_X(X):

    num_img=len(X)

    x_as_array=X.values.reshape(num_img,img_row,img_col,1)

    X_out=x_as_array/255

    return X_out
from keras.utils.np_utils import to_categorical

num_classes=10

def data_prep_Y(Y):

    out_y = to_categorical(Y, num_classes)

    return out_y
X_train = data_prep_X(X_train)

test_data = data_prep_X(test_data)

Y_train = data_prep_Y(Y_train)
# Some examples images

import matplotlib.pyplot as plt

g = plt.imshow(X_train[0][:,:,0])
g = plt.imshow(X_train[1][:,:,0])
from tensorflow.python import keras

from tensorflow.python.keras.models import Sequential

from tensorflow.python.keras.layers import Dense, Flatten, Conv2D



model = Sequential()

model.add(Conv2D(20, kernel_size=(3, 3),

                 activation='relu',

                 input_shape=(img_row, img_col, 1)))

model.add(Conv2D(20, kernel_size=(3, 3), activation='relu'))

model.add(Flatten())

model.add(Dense(128, activation='relu'))

model.add(Dense(num_classes, activation='softmax'))



model.compile(loss=keras.losses.categorical_crossentropy,

              optimizer='adam',

              metrics=['accuracy'])

model.fit(X_train, Y_train,

          batch_size=128,

          epochs=30,

          validation_split = 0.2)
# predict results

results = model.predict(test_data)



# select the indix with the maximum probability

results = np.argmax(results,axis = 1)



results = pd.Series(results,name="Label")
submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)



submission.to_csv("mySubmission.csv",index=False)