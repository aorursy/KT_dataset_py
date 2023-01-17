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
from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix



from keras.datasets import mnist

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D

from keras.optimizers import SGD, Adam, RMSprop

from keras.utils import np_utils



import keras
train_file = "/kaggle/input/digit-recognizer/train.csv"

test_file = "/kaggle/input/digit-recognizer/test.csv"

output_file = "/kaggle/output/digit-recognizer/sample_submission.csv"
raw_data = np.loadtxt(train_file, skiprows=1, dtype='int', delimiter=',')

x_train, x_val, y_train, y_val = train_test_split(raw_data[:,1:], raw_data[:,0], test_size=0.1)
batch_size = 128

num_classes = 10

epochs = 5 



img_rows, img_cols = 28, 28
x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)

x_train = x_train.astype('float32')

x_train /= 255
x_val = x_val.reshape(x_val.shape[0], img_rows, img_cols, 1)

x_val = x_val.astype('float32')

x_val /= 255
input_shape = (img_rows, img_cols, 1)

y_train = keras.utils.to_categorical(y_train, num_classes)

y_val = keras.utils.to_categorical(y_val, num_classes)
model = Sequential()



# первый сверточный слой

model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))



# второй сверточный слой

model.add(Conv2D(64, (3, 3), activation='relu'))



# слой Pooling

model.add(MaxPooling2D(pool_size=(2, 2)))



# слой dropout

model.add(Dropout(0.25))



# растягиваем в вектор

model.add(Flatten())



# первый слой анализа 

model.add(Dense(128, activation='relu'))



# слой dropout

model.add(Dropout(0.5))



# второй слой анализа 

model.add(Dense(num_classes, activation='softmax'))



# определяемся с обучением

model.compile(loss=keras.losses.categorical_crossentropy,  optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])



model.summary()
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_val, y_val))
test= pd.read_csv(test_file)

print(test.shape)

X_test = test.values.astype('float32')

X_test = X_test.reshape(X_test.shape[0], 28, 28,1)
predictions = model.predict_classes(X_test, verbose=0)



submissions=pd.DataFrame({"ImageId": list(range(1,len(predictions)+1)),

                         "Label": predictions})

submissions.to_csv("DR.csv", index=False, header=True)