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
from keras.models import Sequential

from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization

from keras.utils import to_categorical

import matplotlib.pyplot as plt

import pandas as pd

import numpy as np

import warnings

warnings.filterwarnings("ignore")
def load_and_preprocess(data_path):

    data = pd.read_csv(data_path)

    data = data.to_numpy()

    np.random.shuffle(data)

    x = data[:,1:].reshape(-1,28,28,1)/255.0

    y = data[:,0].astype(np.int32)

    y = to_categorical(y, num_classes=len(set(y)))



    return x,y
train_data_path = "/kaggle/input/mnist-in-csv/mnist_train.csv"

test_data_path = "/kaggle/input/mnist-in-csv/mnist_train.csv"



x_train,y_train = load_and_preprocess(train_data_path)

x_test, y_test = load_and_preprocess(test_data_path)
index = 69

vis = x_train.reshape(60000,28,28)

plt.imshow(vis[index,:,:]) 

plt.legend()

plt.axis("off")

plt.show()

print(np.argmax(y_train[index]))
numberOfClass = y_train.shape[1]



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

model.add(Dropout(0.4))

model.add(Dense(units = numberOfClass))

model.add(Activation("softmax"))



model.compile(loss = "categorical_crossentropy",

              optimizer = "adam",

              metrics = ["accuracy"])



hist = model.fit(x_train,y_train, validation_data=(x_test,y_test), epochs= 10, batch_size= 1024)

print(hist.history.keys())

plt.plot(hist.history["loss"],label = "Train Loss")

plt.plot(hist.history["val_loss"],label = "Validation Loss")

plt.legend()

plt.show()
plt.plot(hist.history["accuracy"],label = "Train Accuracy")

plt.plot(hist.history["val_accuracy"],label = "Validation Accuracy")

plt.legend()

plt.show()