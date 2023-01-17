import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import cv2

import numpy as np

import os

from os import listdir

listdir('../input')

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.

from keras.utils import to_categorical

import sys

import os

import keras

import cv2

from keras.preprocessing.image import ImageDataGenerator

from keras import optimizers

from keras.models import Sequential

from keras.layers import Dropout, Flatten, Dense, Activation

from keras.layers.convolutional import Convolution2D, MaxPooling2D, Conv2D, MaxPooling2D

from keras.layers.normalization import BatchNormalization

from keras import callbacks

from keras import backend as K

from keras.optimizers import Adam,SGD

from keras.metrics import categorical_accuracy, top_k_categorical_accuracy, categorical_crossentropy

len(trainarr)
trainarr=[]

trainclass=[]

valarr=[]

valclass=[]

for count,i in enumerate(listdir('../input/trainimage1/')):

    image=cv2.imread('../input/trainimage1/'+i)

    image=cv2.resize(image,(224,224))

    if count<25:

        trainarr.append(image)

        trainclass.append(0)

    else: 

        valarr.append(image)

        valclass.append(0)

for count,i in enumerate(listdir('../input/trainimage2/')):

    image=cv2.imread('../input/trainimage2/'+i)

    image=cv2.resize(image,(224,224))

    if count<25:

        trainarr.append(image)

        trainclass.append(1)

    else: 

        valarr.append(image)

        valclass.append(1)  

        

for count,i in enumerate(listdir('../input/trainimage3/')):

    image=cv2.imread('../input/trainimage3/'+i)

    image=cv2.resize(image,(224,224))

    if count<25:

        trainarr.append(image)

        trainclass.append(2)

    else: 

        valarr.append(image)

        valclass.append(2)    







valarr=np.array(valarr)
trainarr=np.array(trainarr)
for count,i in enumerate(valarr):

    valarr[count]=i/255
for count,i in enumerate(trainarr):

    trainarr[count]=i/255
model = Sequential()



# 1st Convolutional Layer

model.add(Conv2D(filters=96, input_shape=(224, 224, 3), kernel_size=(11, 11), \

                 strides=(4, 4), padding='valid'))

model.add(Activation('relu'))

# Pooling

model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))

# Batch Normalisation before passing it to the next layer

model.add(BatchNormalization())

model.summary()

# 2nd Convolutional Layer

model.add(Conv2D(filters=256, kernel_size=(11, 11), strides=(1, 1), padding='valid'))

model.add(Activation('relu'))

# Pooling

model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2),padding='valid',data_format = 'channels_last'))

# Batch Normalisation

model.add(BatchNormalization())



# 3rd Convolutional Layer

model.add(Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), padding='valid'))

model.add(Activation('relu'))

# Batch Normalisation

model.add(BatchNormalization())



# 4th Convolutional Layer

model.add(Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), padding='valid'))

model.add(Activation('relu'))

# Batch Normalisation

model.add(BatchNormalization())







# Passing it to a dense layer

model.add(Flatten())

# 1st Dense Layer

model.add(Dense(4096, input_shape=(224 * 224 * 3,)))

model.add(Activation('relu'))

# Add Dropout to prevent overfitting

model.add(Dropout(0.4))

# Batch Normalisation

model.add(BatchNormalization())



# 2nd Dense Layer

model.add(Dense(4096))

model.add(Activation('relu'))

# Add Dropout

model.add(Dropout(0.4))

# Batch Normalisation

model.add(BatchNormalization())



# 3rd Dense Layer

model.add(Dense(1000))

model.add(Activation('relu'))

# Add Dropout

model.add(Dropout(0.4))

# Batch Normalisation

model.add(BatchNormalization())



# Output Layer

model.add(Dense(3))

model.add(Activation('softmax'))



model.compile(

    optimizer=Adam(lr=0.0001),

    loss='categorical_crossentropy',

    metrics=[categorical_crossentropy,

             categorical_accuracy])

model.summary()
model.fit(

trainarr,

y,

validation_data=(valarr,z),

batch_size=16,

epochs=10

)
y = to_categorical(trainclass, num_classes=3)
model.predict(valarr[3:5])


from matplotlib.pyplot import imshow

imshow(trainarr[0])
from os import listdir

listdir('../input/trainimage1')
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

from os import listdir

listdir('../input')

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.

import sys

import os

import keras

import cv2

from keras.preprocessing.image import ImageDataGenerator

from keras import optimizers

from keras.models import Sequential

from keras.layers import Dropout, Flatten, Dense, Activation

from keras.layers.convolutional import Convolution2D, MaxPooling2D, Conv2D, MaxPooling2D

from keras.layers.normalization import BatchNormalization

from keras import callbacks

from keras import backend as K

from keras.optimizers import Adam,SGD

from keras.metrics import categorical_accuracy, top_k_categorical_accuracy, categorical_crossentropy









def precision(y_true, y_pred):

    """Precision metric.

    Only computes a batch-wise average of precision.

    Computes the precision, a metric for multi-label classification of

    how many selected items are relevant.

    """

    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))

    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))

    precision = true_positives / (predicted_positives + K.epsilon())

    return precision





def recall(y_true, y_pred):

    """Recall metric.

    Only computes a batch-wise average of recall.

    Computes the recall, a metric for multi-label classification of

    how many relevant items are selected.

    """

    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))

    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))

    recall = true_positives / (possible_positives + K.epsilon())

    return recall





def fbeta_score(y_true, y_pred, beta=1):

    """Computes the F score.

    The F score is the weighted harmonic mean of precision and recall.

    Here it is only computed as a batch-wise average, not globally.

    This is useful for multi-label classification, where input samples can be

    classified as sets of labels. By only using accuracy (precision) a model

    would achieve a perfect score by simply assigning every class to every

    input. In order to avoid this, a metric should penalize incorrect class

    assignments as well (recall). The F-beta score (ranged from 0.0 to 1.0)

    computes this, as a weighted mean of the proportion of correct class

    assignments vs. the proportion of incorrect class assignments.

    With beta = 1, this is equivalent to a F-measure. With beta < 1, assigning

    correct classes becomes more important, and with beta > 1 the metric is

    instead weighted towards penalizing incorrect class assignments.

    """

    if beta < 0:

        raise ValueError('The lowest choosable beta is zero (only precision).')



    # If there are no true positives, fix the F score at 0 like sklearn.

    if K.sum(K.round(K.clip(y_true, 0, 1))) == 0:

        return 0



    p = precision(y_true, y_pred)

    r = recall(y_true, y_pred)

    bb = beta ** 2

    fbeta_score = (1 + bb) * (p * r) / (bb * p + r + K.epsilon())

    return fbeta_score





def fmeasure(y_true, y_pred):

    """Computes the f-measure, the harmonic mean of precision and recall.

    Here it is only computed as a batch-wise average, not globally.

    """

    return fbeta_score(y_true, y_pred, beta=1)





def categorical_crossentropy(y_true, y_pred):

    return K.mean(K.categorical_crossentropy(y_pred, y_true))







"""

Parameters

"""

img_width, img_height = 224, 224

batch_size = 32

samples_per_epoch = 75

validation_steps = 1806

nb_filters1 = 32

nb_filters2 = 64

conv1_size = 3

conv2_size = 2

pool_size = 2

classes_num = 3

lr = 0.0004

decay=0.1

epochs = 2



model = Sequential()



# 1st Convolutional Layer

model.add(Conv2D(filters=96, input_shape=(224, 224, 3), kernel_size=(11, 11), \

                 strides=(4, 4), padding='valid'))

model.add(Activation('relu'))

# Pooling

model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))

# Batch Normalisation before passing it to the next layer

model.add(BatchNormalization())

model.summary()

# 2nd Convolutional Layer

model.add(Conv2D(filters=256, kernel_size=(11, 11), strides=(1, 1), padding='valid'))

model.add(Activation('relu'))

# Pooling

model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2),padding='valid',data_format = 'channels_last'))

# Batch Normalisation

model.add(BatchNormalization())



# 3rd Convolutional Layer

model.add(Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), padding='valid'))

model.add(Activation('relu'))

# Batch Normalisation

model.add(BatchNormalization())



# 4th Convolutional Layer

model.add(Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), padding='valid'))

model.add(Activation('relu'))

# Batch Normalisation

model.add(BatchNormalization())







# Passing it to a dense layer

model.add(Flatten())

# 1st Dense Layer

model.add(Dense(4096, input_shape=(224 * 224 * 3,)))

model.add(Activation('relu'))

# Add Dropout to prevent overfitting

model.add(Dropout(0.4))

# Batch Normalisation

model.add(BatchNormalization())



# 2nd Dense Layer

model.add(Dense(4096))

model.add(Activation('relu'))

# Add Dropout

model.add(Dropout(0.4))

# Batch Normalisation

model.add(BatchNormalization())



# 3rd Dense Layer

model.add(Dense(1000))

model.add(Activation('relu'))

# Add Dropout

model.add(Dropout(0.4))

# Batch Normalisation

model.add(BatchNormalization())



# Output Layer

model.add(Dense(3))

model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',

              optimizer=optimizers.RMSprop(lr=lr),

              metrics=['accuracy', precision, recall, fmeasure])

model.compile(

    optimizer=Adam(lr=0.0001),

    loss='categorical_crossentropy',

    metrics=[categorical_crossentropy,

             categorical_accuracy])

train_datagen = ImageDataGenerator(

    rescale=1. / 255,

    )



test_datagen = ImageDataGenerator(validation_split=0.3,rescale=1. / 255)



train_generator = train_datagen.flow_from_directory(directory=r"../input",

    target_size=(img_height, img_width),

    color_mode='rgb',

    batch_size=batch_size,

    class_mode='categorical',

    subset='training')



validation_generator = test_datagen.flow_from_directory(

    directory=r"../input",

    color_mode='rgb',

    target_size=(img_height, img_width),

    batch_size=batch_size,

    class_mode='categorical',

    subset='validation')



"""

Tensorboard log

"""

log_dir = './tf-log/'

tb_cb = callbacks.TensorBoard(log_dir=log_dir, histogram_freq=0, update_freq='batch')

filepath = './models/weights.{epoch:02d}-{val_loss:.2f}.h5'

checkpoint = keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=False,

                                             save_weights_only=False, mode='auto', period=5)

cbks = [tb_cb, checkpoint]

model.summary()



model.fit_generator(

    train_generator,

    samples_per_epoch=samples_per_epoch,

    epochs=epochs,

    validation_data=validation_generator,

    callbacks=cbks,

    validation_steps=validation_steps)







target_dir = './models/'

if not os.path.exists(target_dir):

    os.mkdir(target_dir)

model.save('./models/model.h5')

model.save_weights('./models/face_detect_weights.h5')
