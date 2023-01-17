import numpy as np # linear algebra



from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix



from tensorflow.python import keras

from tensorflow.python.keras.models import Sequential

from tensorflow.python.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization

#from tensorflow.python.keras.layers import Dense, Flatten, Conv2D, Dropout



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# Prep the data

from keras.utils.np_utils import to_categorical # convert to one-hot-encoding

#from keras.models import Sequential

#from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization

from keras.optimizers import Adam

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import LearningRateScheduler

from IPython.display import Image, display

import matplotlib.pyplot as plt



train_file = "../input/train.csv"

test_file = "../input/test.csv"

output_file = "submission.csv"



img_rows, img_cols = 28, 28



#split data into training and validation sets

raw_data = np.loadtxt(train_file, skiprows=1, dtype='int', delimiter=',')



x_train, x_val, y_train, y_val = train_test_split(raw_data[:,1:], raw_data[:,0], test_size=0.1)

x_train = x_train.reshape(-1, img_rows, img_cols, 1)

x_val = x_val.reshape(-1, img_rows, img_cols, 1)



y_train = to_categorical(y_train)

y_val = to_categorical(y_val)



#an example:

print(x_val.shape)

print(x_train.shape)

plt.axis('off')

plt.imshow(x_train[0].reshape(28,28),cmap='gray')

print(y_train[0])

num_classes = 10

model = Sequential()

model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(img_rows, img_cols, 1)))

model.add(BatchNormalization())

model.add(MaxPool2D(strides=(2,2)))

model.add(Dropout(0.25))



model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', activation ='relu'))

model.add(MaxPool2D(strides=(2,2)))

model.add(Dropout(0.25))



model.add(Conv2D(filters = 64, kernel_size = (3, 3), activation='relu'))

model.add(BatchNormalization())

model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))

model.add(Dropout(0.25))



model.add(Flatten())

model.add(Dense(256, activation = "relu"))

model.add(Dropout(0.5))

model.add(Dense(10, activation = "softmax"))



model.compile(loss=keras.losses.categorical_crossentropy, optimizer='adam', metrics=['accuracy'])



predictions = model.fit(x_train, y_train, batch_size=100, epochs=3, validation_data=(x_val,y_val))



import pandas as pd

test_data = pd.read_csv("../input/test.csv")

test_data = test_data.values.reshape(-1, 28, 28, 1)

print("test shape: ", test_data.shape)

submission_predictions = model.predict_classes(test_data, verbose=0)

print(submission_predictions)

submission_predictions = np.argmax(submission_predictions)

submission_predictions = pd.Series(submission_predictions, name="Label")

submission = pd.concat([pd.Series(range(1 ,28001) ,name = "ImageId"),   submission_predictions],axis = 1)

print(submission)

submission.to_csv("solution.csv",index=False)

#submission.head(10)
