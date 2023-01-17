%matplotlib inline



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pylab as plt

from random import shuffle



import os

import cv2
train_path = "../input/train"



ROWS = 64

COLS = 64

CHANNELS = 3



images = [img for img in os.listdir(train_path)]

images_dog = [img for img in os.listdir(train_path) if "dog" in img]

images_cat = [img for img in os.listdir(train_path) if "cat" in img]



#only taking a subset (less accuracy but faster training)

train_dog = images_dog[:1000]

train_cat = images_cat[:1000]

valid_dog = images_dog[1000:1100]

valid_cat = images_cat[1000:1100]



train_list = train_dog + train_cat

valid_list = valid_dog + valid_cat



shuffle(train_list)



train = np.ndarray(shape=(len(train_list),ROWS, COLS))

train_color = np.ndarray(shape=(len(train_list), ROWS, COLS, CHANNELS), dtype=np.uint8) #, dtype=np.uint8



labels = np.ndarray(len(train_list))



for i, img_path in enumerate(train_list):

    img_color = cv2.imread(os.path.join(train_path, img_path), 1)

    img_color = cv2.resize(img_color, (ROWS, COLS), interpolation=cv2.INTER_CUBIC)

    img = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

    

    train[i] = img

    train_color[i] = img_color

   

    if "dog" in img_path:

        labels[i] = 0

    else:

        labels[i] = 1

    



valid = np.ndarray(shape=(len(valid_list), ROWS, COLS))

valid_color = np.ndarray(shape=(len(valid_list), ROWS, COLS, CHANNELS))

valid_labels = np.ndarray(len(valid_list))



for i, img_path in enumerate(valid_list):

    img_color = cv2.imread(os.path.join(train_path, img_path), 1)

    img_color = cv2.resize(img_color, (ROWS, COLS), interpolation=cv2.INTER_CUBIC)

    img = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

    

    valid[i] = img

    valid_color[i] = img_color

    

    if "dog" in img_path:

        valid_labels[i] = 0

    else:

        valid_labels[i] = 1
n = 33

plt.subplot(1,2,1)

plt.imshow(cv2.imread(os.path.join(train_path, train_list[len(train_list)-n])))

plt.subplot(1,2,2)

plt.imshow(train_color[len(train_list)-n])

plt.title(labels[len(train_list)-n])

plt.show()



for i in range(1):

    plt.imshow(train[i], cmap='gray')

    plt.title(train_list[i])

    plt.show()
def average(data):

    data = data - 127

    data = data/np.max(np.abs(data))

    return data

def average_color(data):

    data = data - 127

    data = data / 255

    return data





#train = average(train)

print(train[1,:,:])

#train_color = average_color(train_color)
from keras.models import Sequential

from keras.layers import Convolution2D, BatchNormalization

from keras.layers import Dense, Activation, Flatten, Dropout, MaxPooling2D

from keras.regularizers import l2

from keras.utils.np_utils import to_categorical

from keras.optimizers import SGD, RMSprop, Adagrad



optimizer = SGD(lr=1e-3, momentum=0.0, decay=0.0, nesterov=False)

optimizer = RMSprop(lr=1e-4)



def convNet():

    model = Sequential()



    model.add(Convolution2D(32, 3, 3, name='conv1',border_mode='same', activation='relu', input_shape=(ROWS, COLS, CHANNELS), dim_ordering='tf'))

    model.add(Convolution2D(32, 3, 3, border_mode='same', activation='relu', dim_ordering='tf'))

    model.add(MaxPooling2D(pool_size=(2,2), border_mode='same', dim_ordering='tf'))





    model.add(Convolution2D(64, 3, 3, border_mode='same', activation='relu', dim_ordering='tf'))

    model.add(Convolution2D(64, 3, 3, border_mode='same', activation='relu', dim_ordering='tf'))

    model.add(MaxPooling2D(pool_size=(2,2), border_mode='same', dim_ordering='tf'))





    model.add(Convolution2D(128, 3, 3, border_mode='same', activation='relu', dim_ordering='tf'))

    model.add(Convolution2D(128, 3, 3, border_mode='same', activation='relu', dim_ordering='tf'))

    model.add(MaxPooling2D(pool_size=(2,2), border_mode='same',dim_ordering='tf'))

    

    model.add(Convolution2D(256, 3, 3, border_mode='same', activation='relu'))

    model.add(Convolution2D(256, 3, 3, border_mode='same', activation='relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))





    model.add(Flatten())



    model.add(Dense(output_dim=256, activation='relu'))

    model.add(Dropout(0.5))



    model.add(Dense(output_dim=256,  activation='relu'))

    model.add(Dropout(0.5))



    model.add(Dense(output_dim=2))  #binary classification

    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    return model



model = convNet()
model.summary()


#model.compile(loss='binary_crossentropy', optimizer=RMSprop(lr=1e-8), metrics=['accuracy'])

labels_ = to_categorical(labels, 2) #convert labels to a matrix representation 

train_ = np.resize(train, (len(train), ROWS, COLS, 1))

#print(train_[2,:,:,0])
n=232

plt.imshow(train_color[n,:,:,:])

plt.title(labels_[n])

print(labels[n])

if "dog" in train_list[n]:

    print("dog")

else:

    print("cat")

print(train_list[n])

plt.show()
hist = model.fit(train_color, labels_, nb_epoch=10, batch_size=16, validation_split=0.2)

print(hist.history)
valid_labels_ = to_categorical(valid_labels, 2)

print(np.count_nonzero(valid_labels_))

#print(valid_labels_)

#valid_ = average(valid)

valid_ = np.resize(valid, (len(valid), ROWS, COLS, 1))
from keras.models import Model

layer_name = 'conv1'



model2 = Model(input=model.input, output=model.get_layer(layer_name).output)



inter_output = model2.predict(valid_color)

print(inter_output.shape)

print(np.count_nonzero(inter_output)/(200.0*32768.0))


pred = model.predict(valid_color)

#print(pred)

print("valid set :", model.evaluate(valid_color, valid_labels_, verbose=False)[1]*100, "%")

print("--------------------")

print('train set :', model.evaluate(train_color, labels_, verbose=False)[1]*100, '%')
