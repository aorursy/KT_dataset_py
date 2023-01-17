# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('../input/ncu-labeled-ionograms-data/Original_Data_1600x800/Test'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np
import time, os
from tensorflow import keras
import glob
import matplotlib.pyplot as plt
path = r'../input/ncu-labeled-ionogram-data/Original_Label_Data_1600x800/HL424_2013313005704.RIQ_p1_r1.npz'
with np.load(path) as npz:
    x = npz['x']
    yIndex = npz['yIndex']
plt.imshow(x)
plt.show()
y = np.zeros((800,1600))
y[yIndex[0], yIndex[1]] = 1
plt.imshow(y, cmap='gray')
plt.show()

print(y.shape)
# start_time = time.time()

# x_train = np.empty((816,256,208,1))
# y_train = np.empty((816,256,208,1))

# # Input
# i = 0
# for dirname, _, filenames in os.walk(r'../input/peru-manual-ionograms/kmeans_816'):
#     for filename in filenames:
#         with np.load(os.path.join(dirname, filename)) as npzdata:
#             power_inp = npzdata['oPower']
#             x_train[i] = np.reshape(power_inp, (256,208,1))
#         i += 1

# # Target
# j = 0
# for dirname, _, filenames in os.walk(r'../input/peru-manual-ionograms/binary_npz_converted'):
#     for filename in filenames:
#         with np.load(os.path.join(dirname, filename)) as npzdata:
#             power_inp = npzdata['oPower']
#             y_train[j] = np.reshape(power_inp, (256,208,1))
#         j += 1


# end_time = time.time()

# # Caculate time
# spend = end_time - start_time
# print(str(spend) + ' s')
# #gen_path = glob.iglob('D:\\Python\\json2npz\\Test\\*.npz')

# zeros = np.zeros((800,1600,11))
# def datagen(inpstr):
#     gen_path = glob.iglob(r'../input/ncu-labeled-ionograms-data/'+ inpstr +'/*.npz')
#     while True:
#         for filename in gen_path:
#             with np.load(filename) as npz:
#                 x = npz['x']
#                 yIndex = npz['yIndex']
#             y = zeros
#             y[yIndex[0], yIndex[1], yIndex[2]] = 1
#             Y_train = y
#             X_train = zeros
#             for i in range(11):
#                 X_train[:,:,i] = x
#             yield X_train, Y_train
# train_set = datagen('Train')
# val_set = datagen('Validation')
# print(train_set)
zeros = np.zeros((800,1600))
def datagen(inpstr):
    while True:
        with open('../input/ncu-labeled-ionogram-data/' + inpstr + '.txt') as f:
            for line in f.readlines():
                with np.load(
                            r'../input/ncu-labeled-ionogram-data/Original_Label_Data_1600x800/'
                            + line.strip() + '.npz'
                            ) as npz:

                    x = npz['x']
                    yIndex = npz['yIndex']
                y = zeros
                y[yIndex[0], yIndex[1]] = 1
                Y_train = np.expand_dims(y, (0, -1))
                X_train = np.expand_dims(x, (0, -1))
                yield X_train, Y_train
            
train_set = datagen('Train_set')
val_set = datagen('Validation_set')
print(train_set)
x, y = next(train_set)
print(x.shape)
print(y.shape)
n = 1000
epoch_times = 10
input_shape = (800, 1600, 1)  # size of the ionogram, expand 1 more
#                            # dimension to fit into 2D CNN
kernel_size = (3, 3) # 3x3 convolution kernel
pool_size = (2, 2) # 2x2 pooling or upsampling

from tensorflow.keras.models import Model

inp = keras.Input(shape=input_shape)               #Input layer merely decides the input shape and is not necessarily a layer.
conv_1 = keras.layers.Conv2D(16, kernel_size=kernel_size, activation='relu',padding='same')(inp)
pool_1 = keras.layers.MaxPooling2D(pool_size=pool_size)(conv_1)
conv_2 = keras.layers.Conv2D(8, kernel_size=kernel_size, activation='relu',padding='same')(pool_1)
pool_2 = keras.layers.MaxPooling2D(pool_size=pool_size)(conv_2)
conv_3 = keras.layers.Conv2D(8, kernel_size=kernel_size, activation='relu',padding='same')(pool_2)
pool_3 = keras.layers.MaxPooling2D(pool_size=pool_size)(conv_3)
conv_4 = keras.layers.Conv2D(8, kernel_size=kernel_size, activation='relu',padding='same')(pool_3)
up_1 = keras.layers.UpSampling2D(size=pool_size)(conv_4)
conv_5= keras.layers.Conv2D(8, kernel_size=kernel_size, activation='relu',padding='same')(up_1)
up_2 = keras.layers.UpSampling2D(size=pool_size)(conv_5)
conv_6 = keras.layers.Conv2D(16, kernel_size=kernel_size, activation='relu',padding='same')(up_2)
up_3 = keras.layers.UpSampling2D(size=pool_size)(conv_6)
conv_7 = keras.layers.Conv2D(1, kernel_size=kernel_size, activation='relu',padding='same')(up_3)
#autoenc = keras.Sequential([inp, conv_1, pool_1, conv_2, pool_2,conv_3, up_1, conv_4, up_2, conv_7])
autoenc = Model(inputs=inp, outputs=conv_7)
autoenc.summary()




# Choose the loss function, optimizer and the metrics for the training.
autoenc.compile(loss='binary_crossentropy', optimizer='adadelta', metrics=[keras.metrics.MeanIoU(num_classes=2)])

# Train the model serveral epochs, with batch size of 128 samples.
# autoenc.fit(train_set, validation_data=val_set, epochs=epoch_times, batch_size=32)

autoenc.fit(train_set)
