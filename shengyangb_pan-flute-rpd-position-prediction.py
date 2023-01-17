# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))





# Any results you write to the current directory are saved as output.



from sklearn.preprocessing import normalize

from keras import backend as K

from random import shuffle

K.tensorflow_backend._get_available_gpus()



training_set = np.load("/kaggle/input/25tev-neutrons-uniplane44-perpanflute-sum40k/2.5TeV_neutrons_Uniplane44_Perpanflute_sum40k.npy", allow_pickle = True)

validate_set = np.load("/kaggle/input/2.5tev_neutrons_uniplane22_perpanflute_sum30k/2.5TeV_neutrons_Uniplane22_Perpanflute_sum30k.npy", allow_pickle = True)

training_set.shape

def get_target_and_feature(ndarray, normalized, pad = 0):

        img = []

        for i in range(ndarray[0].shape[0]):

            if normalized is True:

                tmp = ndarray[0][i] / max(ndarray[0][i])

            else:

                tmp = ndarray[0][i]

            img.append(np.flip(tmp.reshape((4,4)).T,1))

        img = np.array(img)

        target = [np.around(ndarray[1].astype(float),decimals = 2), np.around(ndarray[2].astype(float),decimals = 2)]

        feature = np.array(img)

        feature_array = feature.reshape((-1,4,4,1))

        if pad != 0:

            feature_array = np.pad(feature_array[:, :, :, :], ((0, 0), (pad, pad), (pad, pad), (0,0)), 'constant')

        target_array = convert_target_array(target)

        return feature_array, target_array



def convert_target_array(target):

    target_array = []

    for i in range(len(target[0])):

        target_array.append([target[0][i], target[1][i]])

    return np.array(target_array)



def compare_truth_and_predict(model, true_target, X, idx = 316):

    truth = true_target[idx]

    pred = model.predict(X[idx].reshape(1,4,4,1))[0]

    print("truth", truth)

    print("predict",pred)



def convert_pos_to_tile(pos):

    if pos <= -20:

        return 0

    elif pos > -20 and pos <= 0:

        return 1

    elif pos > 0 and pos <= 20:

        return 2

    elif pos > 20 and pos <= 40:

        return 3

    

def categorize_position(pos):

    X_tile = convert_pos_to_tile(pos[0])

    Y_tile = convert_pos_to_tile(pos[1])

    tmp = np.zeros((4,4))

    tmp[X_tile, Y_tile] = 1

    return np.array(tmp.reshape((16)))



def plot_feature_and_target(feature_array, target_array, model = None, idx = 316):

    print("Truth:", target_array[idx][0], target_array[idx][1])

    if model is not None:

        print("Prediction:", model.predict(feature_array[idx].reshape(1,4,4,1))[0] )

    plt.imshow(feature_array[idx].reshape((4,4)))

    #print(feature_array[idx])

    plt.colorbar()
raw_feature_array, raw_target_array = get_target_and_feature(validate_set, True, 1)
feature_array, target_array = get_target_and_feature(training_set, True, 1)

from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(feature_array, target_array, test_size=0.1, random_state=42)
X_train.shape
# Set the CNN model 

# my CNN architechture is In -> [[Conv2D->relu]*2 -> MaxPool2D -> Dropout]*2 -> Flatten -> Dense -> Dropout -> Out

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D



# import regularizer

from keras.regularizers import l1



model = Sequential()



model.add(Conv2D(filters = 16, kernel_size = (2,2),padding = 'Same', 

                 activation ='relu', input_shape = (6,6,1)))

model.add(Conv2D(filters = 16, kernel_size = (2,2),padding = 'Same', 

                 activation ='relu'))

#model.add(MaxPool2D(pool_size=(2,2)))

#model.add(Dropout(0.25))





model.add(Conv2D(filters = 32, kernel_size = (2,2),padding = 'Same', 

                 activation ='relu'))

model.add(Conv2D(filters = 32, kernel_size = (2,2),padding = 'Same', 

                 activation ='relu'))

#model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))

model.add(Dropout(0.25))





model.add(Flatten())

model.add(Dense(50, activation = "relu"))

model.add(Dense(50, activation = "relu"))



model.add(Dense(2, activation = "linear"))
#from keras.optimizers import RMSprop

#optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

model.compile(optimizer='adam', loss = "mse", metrics=["mae"])
#train the model

model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs= 1000,  batch_size=1000)

# validate set

"""

Vfeature, Vtarget = get_target_and_feature(validate_set, True)

prediction = model.predict(Vfeature)

idx = 3

print(f"ML prediction {prediction[idx]}")

print(f"Truth {Vtarget[idx]}")

plt.imshow(Vfeature[idx].reshape(4,4))

"""
model.save("2.5TeV_neutrons_Uniplane44_Perpanflute_sum40k_MSEloss_padded.h5")