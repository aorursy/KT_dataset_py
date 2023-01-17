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

import warnings

warnings.filterwarnings("ignore")

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from keras.models import Sequential

from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense,BatchNormalization

from keras.utils import to_categorical

from keras.preprocessing.image  import ImageDataGenerator, img_to_array,load_img

import matplotlib.pyplot as plt

from glob import glob
def load_and_preprocess(data_path):

    data = pd.read_csv(data_path)

    data = data.to_numpy()

    np.random.shuffle(data)

    x = data[:,1:].reshape(-1,28,28,1)/255.0

    y = data[:,0].astype(np.int32)

    y = to_categorical(y, num_classes=len(set(y)))



    return x,y
x_train,y_train = load_and_preprocess("/kaggle/input/mnist-in-csv/mnist_train.csv"

)

print("Shape of x_train : " , x_train.shape)

print("Shape of y_train : " , y_train.shape)
x_test,y_test = load_and_preprocess("/kaggle/input/mnist-in-csv/mnist_test.csv"

)

print("Shape of x_test : " , x_test.shape)

print("Shape of y_test : " , y_test.shape)
i = 20;

temp = x_train.reshape(60000,28,28)

plt.imshow(temp[i,:,:])

plt.legend()

plt.axis('off')

plt.title(np.argmax(y_train[i]))

plt.show()
number_of_class = y_train.shape[1]
model = Sequential()
model.add(Conv2D(input_shape = (28,28,1), filters = 16, kernel_size = (3,3)))

model.add(BatchNormalization())

model.add(Activation("relu"))

model.add(MaxPooling2D())
model.add(Conv2D(filters = 64, kernel_size = (3,3)))

model.add(BatchNormalization())

model.add(Activation("relu"))

model.add(MaxPooling2D())
model.add(Conv2D(filters = 128, kernel_size = (3,3)))

model.add(BatchNormalization())

model.add(Activation("relu"))

model.add(MaxPooling2D())
model.add(Flatten())

model.add(Dense(units = 256))

model.add(Activation("relu"))
model.add(Dropout(0.2))

model.add(Dense(units = number_of_class))

model.add(Activation("softmax"))
model.compile(loss = "categorical_crossentropy",

              optimizer = "adam",

              metrics = ["accuracy"])
hist = model.fit(x_train,y_train, validation_data=(x_test,y_test), epochs= 5, batch_size= 32)
model.save_weights('cnn_mnist_model.h5')
print(hist.history.keys())

plt.plot(hist.history["loss"],label = "Train Loss")

plt.plot(hist.history["val_loss"],label = "Validation Loss")

plt.legend()

plt.show()

plt.figure()

plt.plot(hist.history["accuracy"],label = "Train Accuracy")

plt.plot(hist.history["val_accuracy"],label = "Validation Accuracy")

plt.legend()

plt.show()
#%% save history

import json

with open('cnn_mnist_hist.json', 'w') as f:

    json.dump(hist.history, f)

    

#%% load history

import codecs

with codecs.open("cnn_mnist_hist.json", 'r', encoding='utf-8') as f:

    h = json.loads(f.read())



plt.figure()

plt.plot(h["loss"],label = "Train Loss")

plt.plot(h["val_loss"],label = "Validation Loss")

plt.legend()

plt.show()

plt.figure()

plt.plot(h["accuracy"],label = "Train Accuracy")

plt.plot(h["val_accuracy"],label = "Validation Accuracy")

plt.legend()

plt.show()