import os, cv2, random

import numpy as np

import pandas as pd
import theano
from keras import backend as K

K.image_dim_ordering()
K.set_image_dim_ordering('th')
import matplotlib.pyplot as plt

from matplotlib import ticker

import seaborn as sns

%matplotlib inline 



from keras.models import Sequential

from keras.layers import Input, Dropout, Flatten, Convolution2D, MaxPooling2D, Dense, Activation

from keras.optimizers import RMSprop, SGD, Adam, adadelta

from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping

from keras.utils import np_utils
TRAIN_DIR = '../input/train/'

TEST_DIR = '../input/test/'



ROWS = 64

COLS = 64

CHANNELS = 3



#train_images = [TRAIN_DIR+i for i in os.listdir(TRAIN_DIR)] # use this for full dataset

train_dogs =   [TRAIN_DIR+i for i in os.listdir(TRAIN_DIR) if 'dog' in i]

train_cats =   [TRAIN_DIR+i for i in os.listdir(TRAIN_DIR) if 'cat' in i]



test_images =  [TEST_DIR+i for i in os.listdir(TEST_DIR)]





# slice datasets for memory efficiency on Kaggle Kernels, delete if using full dataset

train_images = train_dogs[:10] + train_cats[:10]

random.shuffle(train_images)

test_images =  test_images[:10]



def read_image(file_path):

    

    img=cv2.resize(cv2.imread(file_path), (ROWS, COLS), interpolation=cv2.INTER_CUBIC)

    img=img.transpose((2,0,1)) #use this if you want the image to be (channel, row, cols)

    return img



def prep_data(images):

    count = len(images)

    data = np.ndarray((count,CHANNELS, ROWS, COLS ))



    for i, image_file in enumerate(images):

        image = read_image(image_file)

        data[i] = image

        if i%250 == 0: print('Processed {} of {}'.format(i, count))

    

    return data



train = prep_data(train_images)

test = prep_data(test_images)



print("Train shape: {}".format(train.shape))

print("Test shape: {}".format(test.shape))
labels = []

for i in train_images:

    if 'dog' in i:

        labels.append(1)

    else:

        labels.append(0)

        

labels_test = []

for i in test_images:

    if 'dog' in i:

        labels_test.append(1)

    else:

        labels_test.append(0)

        
optimizer = RMSprop(1e-4)

#logloss-> binary_crossentropy

objective = 'binary_crossentropy'



#set dim_ordering to 'th' so it reads channel dimension at index 1

def catdog():

    

    model = Sequential()



    model.add(Convolution2D(32, 3, 3, border_mode='same', input_shape=(3, ROWS, COLS),activation='relu'))

    model.add(Convolution2D(32, 3, 3, border_mode='same' ,activation='relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))



    model.add(Convolution2D(64, 3, 3, border_mode='same', activation='relu'))

    model.add(Convolution2D(64, 3, 3, border_mode='same', activation='relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    

    model.add(Convolution2D(128, 3, 3, border_mode='same', activation='relu'))

    model.add(Convolution2D(128, 3, 3, border_mode='same', activation='relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    

    model.add(Convolution2D(256, 3, 3, border_mode='same', activation='relu'))

    model.add(Convolution2D(256, 3, 3, border_mode='same',activation='relu'))

#     model.add(Convolution2D(256, 3, 3, border_mode='same', activation='relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))



#     model.add(Convolution2D(256, 3, 3, border_mode='same', activation='relu'))

#     model.add(Convolution2D(256, 3, 3, border_mode='same', activation='relu'))

#     model.add(Convolution2D(256, 3, 3, border_mode='same', activation='relu'))

#     model.add(MaxPooling2D(pool_size=(2, 2)))



    model.add(Flatten())

    model.add(Dense(256, activation='relu'))

    model.add(Dropout(0.5))

    

    model.add(Dense(256, activation='relu'))

    model.add(Dropout(0.5))



    model.add(Dense(1))

    model.add(Activation('sigmoid'))

    

    model.compile(loss=objective, optimizer=optimizer, metrics=['accuracy'])

    

    return model





model = catdog()
nb_epoch = 3

batch_size = 5



from keras.callbacks import History 

history = History()

early_stopping=EarlyStopping(monitor='val_loss', patience=1)

train_model=model.fit(train, labels, batch_size=batch_size, nb_epoch=nb_epoch,validation_split=0.1, verbose=1, shuffle=True, callbacks=[history, early_stopping])

    
# train the model on the new data for a few epochs
predictions = model.predict(test, verbose=0)

print(predictions)
np.mean(train_model.history['val_loss'])
plt.plot(train_model.history['loss'], 'blue', label='Training Loss')

plt.plot(train_model.history['val_loss'], 'green', label='Validation Loss')

plt.xlabel('epoch')

plt.ylabel('log loss')

plt.legend(loc='upper right')

plt.show()
for i in range(0,10):

    if predictions[i, 0] >= 0.5: 

        print('I am {:.2%} sure this is a Dog'.format(predictions[i][0]))

    else: 

        print('I am {:.2%} sure this is a Cat'.format(1-predictions[i][0]))

        

    plt.imshow(test[i].T)

    plt.show()
from keras.preprocessing.image import ImageDataGenerator
datagen = ImageDataGenerator(

    featurewise_center=True,

    featurewise_std_normalization=True,

    horizontal_flip=True)


# compute quantities required for featurewise normalization

# (std, mean, and principal components if ZCA whitening is applied)

datagen.fit(train)
X_train = train.reshape(train.shape[0], 3, 64, 64)
x = img_to_array(img)

x = x.reshape((1,) + x.shape)
X= datagen.flow(train, labels, batch_size=5)
X
# fits the model on batches with real-time data augmentation:

model.fit_generator(X,

                    samples_per_epoch=train.shape[0],nb_epoch=nb_epoch,verbose=2)

                    #validation_data=(test[0:5], labels_test[0:5]),

                    #callbacks = [history])