# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

%matplotlib notebook

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
from keras.models import Sequential

from keras.layers import Dense, Dropout, Activation, Conv2D,  MaxPool2D, Flatten

from keras.utils import to_categorical



from sklearn.model_selection import train_test_split
use_complete_dataset = False  # Use the total dataset, or just a sample of it for quick runs.



if use_complete_dataset:

    train_data = pd.read_csv("../input/train.csv")

    test_data = pd.read_csv("../input/test.csv")

else:

    train_data = pd.read_csv("../input/train.csv", nrows=10000)

    test_data = pd.read_csv("../input/test.csv")

    

# Perform quick sanity check

train_data.head(5)
x = train_data.iloc[:,1:].values.astype("float32")

y = train_data.iloc[:,0].values.astype("int32")



x_test = test_data.values.astype("float32")
image_size = 28

num_classes = 10

max_value = 255.0



# Rescale the pixel values to the range [0:1]

x = x/max_value

x_test = x_test/max_value



# Reshape the images (to match the Keras format)

x = x.reshape(x.shape[0], image_size, image_size, 1)

x_test = x_test.reshape(x_test.shape[0], image_size, image_size, 1)



# Convert the labels to categorical (for the keras classifier)

y = to_categorical(y, num_classes= num_classes)



# Plot the images as sanity check

plt.figure()

plt.imshow(x[0,:,:,0])

plt.figure()

plt.imshow(x_test[0,:,:,0])
x_train, x_validation, y_train, y_validation = train_test_split(x, y, test_size = 0.2) 
model = Sequential()

model.add(Conv2D(64, kernel_size=(3,3), activation = "relu", data_format = "channels_last" ,input_shape=(image_size, image_size, 1)))

model.add(MaxPool2D(pool_size=(2,2)))

model.add(Conv2D(128, kernel_size=(2,2), activation = "relu"))

model.add(MaxPool2D(pool_size=(2,2)))

model.add(Flatten())

model.add(Dense(128, activation="relu"))

model.add(Dropout(0.25))

model.add(Dense(num_classes, activation="softmax"))



model.summary()
model.compile(loss = "categorical_crossentropy", optimizer = "adam", metrics = ["accuracy"])
model.fit(x_train, y_train, batch_size = 32, epochs= 10)
model.evaluate(x_validation,y_validation, batch_size=32)
labels_predicted = np.argmax(model.predict(x_test, batch_size=32), axis=1)



# Write the output file

predicted_data = pd.DataFrame({'ImageId':np.arange(1,labels_predicted.size +1 ),'Label':labels_predicted})

predicted_data.to_csv("mnist_submission_matgarate.csv", index=False)
# Check the predictions for one of the images

i = 15

print("Label Predicted:  "  + str(int(labels_predicted[i])))

plt.figure()

plt.title("Number")

plt.imshow(x_test[i].reshape(28,28))