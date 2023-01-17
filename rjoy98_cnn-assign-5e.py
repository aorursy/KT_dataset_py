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
import glob
import os
import cv2
import pandas as pd
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from random import shuffle

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dropout
from keras.layers import Flatten
from keras.utils import np_utils
import matplotlib.pyplot as plt
import gc

gc.enable()
# read names of all flowers 
file_names = {'daisy' : [], 
              'dandelion' : [], 
              'rose' : [], 
              'sunflower' : [], 
              'tulip' : []}

images = {'daisy' : [], 
          'dandelion' : [], 
          'rose' : [], 
          'sunflower' : [], 
          'tulip' : []}

flower_types = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']
for k in range(len(flower_types)):
    names = glob.glob("../input/flowers/flowers/" + str(flower_types[k]) + "/*.jpg")
    print (flower_types[k], ":", len(names))
    file_names[str(flower_types[k])] = names
for k in range(len(flower_types)):
    data_ = []
    names = file_names[str(flower_types[k])]
    #directory = "/content/gdrive/My Drive/assignment_5_cnn/data/flowers/" + str(flower_types[k]) + "/"
    for m in tqdm(range(len(names))):
        path = str(names[m])
        loaded_image = cv2.imread(path, cv2.IMREAD_UNCHANGED) #cv2.IMREAD_UNCHANGED #cv2.IMREAD_GRAYSCALE
        loaded_image = cv2.resize(loaded_image, (120, 120)).reshape(120,120,3)
        loaded_image = loaded_image * 1.0
        loaded_image = loaded_image / 255.0
        data_.append(loaded_image)
    #data_ = data_[:700]
    images[str(flower_types[k])] = data_
x = []
y = []

for k in tqdm(range(len(flower_types))):
    x_data = images[str(flower_types[k])]
    y_data = [0] * len(flower_types)
    y_data[k] = 1
    for m in range(len(x_data)):
        x.append(x_data[m])
        y.append(y_data)

print ("\nx :", len(x))
print ("y :", len(y))
total_data = {"data" : x, "label" : y}
total_data = pd.DataFrame(total_data)
total_data = total_data.as_matrix()
total_data = np.array(total_data)
np.random.shuffle(total_data)
data_x = []
data_y = []

for k in tqdm(range(len(total_data))):
    data_x.append(total_data[k][0])
    data_y.append(total_data[k][1])
trainX = data_x[:int(len(data_x) * 0.8)]
trainY = data_y[:int(len(data_y) * 0.8)]
validateX = data_x[int(len(data_x) * 0.8) : int(len(data_x) * 0.9)]
validateY = data_y[int(len(data_y) * 0.8) : int(len(data_y) * 0.9)]
testX = data_x[int(len(data_x) * 0.9):]
testY = data_y[int(len(data_y) * 0.9):]

trainX = np.array(trainX)
trainY = np.array(trainY)
validateX = np.array(validateX)
validateY = np.array(validateY)
testX = np.array(testX)
testY = np.array(testY)


print (len(trainX))
print (len(validateX))
print (len(testX))
import glob
import os
import cv2
import pandas as pd
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from random import shuffle

import keras
from keras.models import Sequential
from keras.models import Model
from keras.layers import Dense
from keras.layers import Conv1D
from keras.layers import Conv2D
from keras.layers import MaxPooling1D
from keras.layers import MaxPooling2D
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Input
from keras.layers import Reshape
from keras.utils import np_utils
import matplotlib.pyplot as plt
from keras.layers import Activation, Dense, BatchNormalization

from keras.utils import plot_model
input_shape = Input(shape=(120, 120, 3))

layer1 = Conv2D(32, kernel_size = (5, 5), strides = (1, 1), padding = 'same')(input_shape)
#batch1 = BatchNormalization()(layer1)
activate1 = Activation('relu')(layer1)
maxpl1 = MaxPooling2D(pool_size=(3, 3), strides = (2,2), padding = 'valid')(activate1)

layer2 = Conv2D(64, kernel_size = (5, 5), strides = (1, 1), padding = 'same')(maxpl1)
#batch2 = BatchNormalization()(layer2)
activate2 = Activation('relu')(layer2)
maxpl2 = MaxPooling2D(pool_size=(3, 3), strides = (2,2), padding = 'valid')(activate2)

fl1 = Flatten()(maxpl2)

dense1 = Dense(1000, activation = 'relu')(fl1)
dense2 = Dense(128, activation = 'relu')(dense1)
out = Dense(5, activation = 'softmax')(dense2)

model = Model(input_shape, out)
model.summary()
#model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#model.compile(loss=keras.losses.categorical_crossentropy, optimizer='adam', metrics=['accuracy'])
model.compile(loss=keras.losses.categorical_crossentropy, optimizer='adam', metrics=['accuracy'])
#history = model.fit(trainX, trainY, epochs=4, batch_size=1000, verbose=1, validation_data = (validateX, validateY))
history = model.fit(trainX, trainY, batch_size = 100, epochs = 10, verbose=2, validation_data = (validateX, validateY))
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validate'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validate'], loc='upper left')
plt.show()
scores_train = model.evaluate(validateX, validateY)
print("Validate", "%s: %.2f%%" % (model.metrics_names[1], scores_train[1]*100))

scores_train = model.evaluate(testX, testY)
print("Test", "%s: %.2f%%" % (model.metrics_names[1], scores_train[1]*100))
def name(d):
    maxx = max(d)
    pos = 0
    for k in range(len(d)):
        if maxx == d[k]:
            pos = k
    return flower_types[pos]
d = testX[0]
d = d * 255.0
d = np.int_(d)
plt.imshow(d)
print (name(trainY[0]))
d = testX[1]
d = d * 255.0
d = np.int_(d)
plt.imshow(d)
print (name(trainY[1]))
d = testX[2]
d = d * 255.0
d = np.int_(d)
plt.imshow(d)
print (name(trainY[2]))
d = testX[3]
d = d * 255.0
d = np.int_(d)
plt.imshow(d)
print (name(trainY[3]))
d = testX[4]
d = d * 255.0
d = np.int_(d)
plt.imshow(d)
print (name(trainY[4]))
d = testX[5]
d = d * 255.0
d = np.int_(d)
plt.imshow(d)
print (name(trainY[5]))
d = testX[6]
d = d * 255.0
d = np.int_(d)
plt.imshow(d)
print (name(trainY[6]))
d = testX[7]
d = d * 255.0
d = np.int_(d)
plt.imshow(d)
print (name(trainY[7]))
d = testX[8]
d = d * 255.0
d = np.int_(d)
plt.imshow(d)
print (name(trainY[8]))
d = testX[9]
d = d * 255.0
d = np.int_(d)
plt.imshow(d)
print (name(trainY[9]))
