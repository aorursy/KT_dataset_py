import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from keras.layers import Input, Dense, Conv2D, Flatten

from keras.models import Model

from keras.optimizers import adam

import matplotlib.pyplot as plt

import keras



%matplotlib inline



print(keras.__version__)

print(keras.backend.image_data_format())



csv_test = pd.read_csv('../input/test.csv')

test_images = csv_test.iloc[:, :]

X_test = test_images.values.reshape((test_images.values.shape[0], 28, 28, 1)).astype('int16')



csv = pd.read_csv('../input/train.csv')

images = csv.iloc[:,1:]

labels = csv.iloc[:,:1]



X_train = images.values.reshape((images.values.shape[0], 28, 28, 1)).astype('int16')

Y_train = np.zeros(shape=(labels.shape[0], 10), dtype='int8')

for i in range(labels.shape[0]):

    Y_train[i][labels.values[i][0]] = 1



# This returns a tensor

input = Input(shape=(28, 28, 1))



# a layer instance is callable on a tensor, and returns a tensor

x = Conv2D(6, kernel_size=(3, 3), activation='relu', input_shape=(28, 28))(input)

x = Conv2D(4, kernel_size=(3, 3), activation='relu')(x)

x = Flatten()(x)

x = Dense(128, activation='linear')(x)

x = Dense(64, activation='tanh')(x)

predictions = Dense(10, activation='softmax')(x)



# This creates a model that includes

# the Input layer and three Dense layers

model = Model(inputs=input, outputs=predictions)

model.compile(optimizer=adam(lr=0.0001), loss='categorical_crossentropy')
#model.optimizer.lr.set_value(0.0001)

model.fit(x=X_train, y=Y_train, epochs=10, validation_split=0.1)  # starts training
model.optimizer.lr.set_value(0.00001)

model.fit(x=X_train, y=Y_train, epochs=10, validation_split=0.1)  # starts training
P_test = model.predict(x=X_test)

Data = np.zeros(shape=(P_test.shape[0], 2)).astype('int32')

for i in range(P_test.shape[0]):

    Data[i][0] = i + 1

    Data[i][1] = np.argmax(P_test[i])

np.savetxt('../input/submission.csv', Data, fmt='%d', delimiter=',', header='ImageId,Label', comments='')
for i in range(30):

    print('Label: ', Data[i])

    a = X_test[i]

    plt.imshow(a.reshape((28, 28)), cmap='gray')

    plt.show()