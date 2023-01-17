# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from pprint import pprint





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

root_train_directory = '/kaggle/input/human-protein-atlas-image-classification/train/'

root_test_directory = '/kaggle/input/human-protein-atlas-image-classification/test/'

input_shape=(128,128,4)





# Any results you write to the current directory are saved as output.



name_label_dict = {

0:  'Nucleoplasm',

1:  'Nuclear membrane',

2:  'Nucleoli',   

3:  'Nucleoli fibrillar center',

4:  'Nuclear speckles',

5:  'Nuclear bodies',

6:  'Endoplasmic reticulum',   

7:  'Golgi apparatus',

8:  'Peroxisomes',

9:  'Endosomes',

10:  'Lysosomes',

11:  'Intermediate filaments',

12:  'Actin filaments',

13:  'Focal adhesion sites',   

14:  'Microtubules',

15:  'Microtubule ends',  

16:  'Cytokinetic bridge',   

17:  'Mitotic spindle',

18:  'Microtubule organizing center',  

19:  'Centrosome',

20:  'Lipid droplets',

21:  'Plasma membrane',   

22:  'Cell junctions', 

23:  'Mitochondria',

24:  'Aggresome',

25:  'Cytosol',

26:  'Cytoplasmic bodies',   

27:  'Rods & rings' }
train_df = pd.read_csv('/kaggle/input/human-protein-atlas-image-classification/train.csv')

train_df.set_index('Id',inplace=True)

train = train_df.to_dict()['Target']

train_img_names = list(train.keys())
SIZE = (128,128)

def get_labels(img):

    return list(map(int, train[img].split(' ')))

# We'll fusion images

def open_multilayer_image(path, test=False):

    fullpath = root_train_directory+path

    if test:

        fullpath = root_test_directory+path

    red = plt.imread(fullpath+"_red.png")

    red = cv2.resize(red, SIZE)

    green = plt.imread(fullpath+"_green.png")

    green = cv2.resize(green, SIZE)

    blue = plt.imread(fullpath+"_blue.png")

    blue = cv2.resize(blue, SIZE)

    yellow = plt.imread(fullpath+"_yellow.png")

    yellow = cv2.resize(yellow, SIZE)

    ni = np.zeros((SIZE[0],SIZE[1],4), 'uint8')

    ni[..., 0] = red*255

    ni[..., 1] = green*255

    ni[..., 2] = blue*255

    ni[..., 3] = yellow*255

    return ni
import cv2

import matplotlib.pyplot as plt

from random import randrange

from textwrap import wrap



#print('labels:',get_labels(img_name))



fig=plt.figure(figsize=(10, 16))



columns = 4

rows = 5

for i in range(1, columns*rows +1):

    num = randrange(len(train_img_names))

    img_name = train_img_names[num]

    img = open_multilayer_image(img_name)

    sub = fig.add_subplot(rows, columns, i)

    # make title

    title = ''

    for label in get_labels(img_name):

        title+=name_label_dict[label]+', '

    sub.set_title("\n".join(wrap(title[:-2],25)))

    plt.axis('off')

    plt.imshow(img[...,:-1])

plt.show();
from __future__ import print_function

import keras

from keras.datasets import mnist

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten

from keras.layers import Conv2D, MaxPooling2D

from keras import backend as K

from keras.callbacks import EarlyStopping,ModelCheckpoint



batch_size = 128

num_classes = len(name_label_dict)

epochs = 60



def preprocess_x(share, train_img_names, test=False):

    x_train = []

    for i in range(0,int(len(train_img_names)*share)):

        x_train.append(open_multilayer_image(train_img_names[i], test))

    x_train = np.array(x_train)

    x_train = x_train.astype('float32')

    x_train /= 255

    return x_train





def preprocess_y(share, train_img_names):

    y_train = []

    for i in range(0,int(len(train_img_names)*share)):

        y_train.append(get_labels(train_img_names[i]))

    y_train = np.array(y_train)



    # convert class vectors to binary class matrices

    y_train_formatted = []

    for y in y_train:

        label = np.zeros(num_classes)

        for j in y:

            label[j]=1

        y_train_formatted.append(label)

    y_train = np.array(y_train_formatted)

    return y_train





x_train  = preprocess_x(0.01, train_img_names)

y_train  = preprocess_y(0.01, train_img_names)

# Create a test set

x_test = x_train[int(len(x_train)*0.8):]

x_train = x_train[:int(len(x_train)*0.8)]



y_test = y_train[int(len(y_train)*0.8):]

y_train = y_train[:int(len(y_train)*0.8)]

# Data genetor



import numpy as np

import cv2

from tensorflow.keras.utils import Sequence





class DataGenerator(Sequence):

    """Generates data for Keras

    Sequence based data generator. Suitable for building data generator for training and prediction.

    """

    def __init__(self, list_IDs, image_path,

                 to_fit=True, batch_size=32, dim=SIZE,

                 n_channels=4, n_classes=10, shuffle=True):

        """Initialization

        :param list_IDs: list of all 'label' ids to use in the generator

        :param image_path: path to images location

        :param to_fit: True to return X and y, False to return X only

        :param batch_size: batch size at each iteration

        :param dim: tuple indicating image dimension

        :param n_channels: number of image channels

        :param n_classes: number of output masks

        :param shuffle: True to shuffle label indexes after every epoch

        """

        self.list_IDs = list_IDs

        self.image_path = image_path

        self.to_fit = to_fit

        self.batch_size = batch_size

        self.dim = dim

        self.n_channels = n_channels

        self.n_classes = n_classes

        self.shuffle = shuffle

        self.on_epoch_end()



    def __len__(self):

        """Denotes the number of batches per epoch

        :return: number of batches per epoch

        """

        return int(np.floor(len(self.list_IDs) / self.batch_size))



    def __getitem__(self, index):

        """Generate one batch of data

        :param index: index of the batch

        :return: X and y when fitting. X only when predicting

        """

        # Generate indexes of the batch

        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]



        # Find list of IDs

        list_IDs_temp = [self.list_IDs[k] for k in indexes]



        # Generate data

        X = self._generate_X(list_IDs_temp)



        if self.to_fit:

            y = self._generate_y(list_IDs_temp)

            return X, y

        else:

            return X



    def on_epoch_end(self):

        """Updates indexes after each epoch

        """

        self.indexes = np.arange(len(self.list_IDs))

        if self.shuffle == True:

            np.random.shuffle(self.indexes)



    def _generate_X(self, list_IDs_temp):

        """Generates data containing batch_size images

        :param list_IDs_temp: list of label ids to load

        :return: batch of images

        """

        # Initialization

        X = np.empty((self.batch_size, *self.dim, self.n_channels))



        # Generate data

        for i, ID in enumerate(list_IDs_temp):

            # Store sample

            X[i,] = self._load_and_preprocess(self.image_path + ID)



        return X



    def _generate_y(self, list_IDs_temp):

        """Generates data containing batch_size masks

        :param list_IDs_temp: list of label ids to load

        :return: batch if masks

        """

        y_train = []

        for img_id in list_IDs_temp:

            y_train.append(get_labels(img_id))

        y_train = np.array(y_train)



        # convert class vectors to binary class matrices

        y_train_formatted = []

        for y in y_train:

            label = np.zeros(num_classes)

            for j in y:

                label[j]=1

            y_train_formatted.append(label)

        y_train = np.array(y_train_formatted)

        return y_train



    def _load_and_preprocess(self, image_path):

        """Load grayscale image

        :param image_path: path to image to load

        :return: loaded image

        """

        fullpath = image_path

        red = plt.imread(fullpath+"_red.png")

        red = cv2.resize(red, SIZE)

        green = plt.imread(fullpath+"_green.png")

        green = cv2.resize(green, SIZE)

        blue = plt.imread(fullpath+"_blue.png")

        blue = cv2.resize(blue, SIZE)

        yellow = plt.imread(fullpath+"_yellow.png")

        yellow = cv2.resize(yellow, SIZE)

        ni = np.zeros((SIZE[0],SIZE[1],4), 'uint8')

        ni[..., 0] = red*255

        ni[..., 1] = green*255

        ni[..., 2] = blue*255

        ni[..., 3] = yellow*255

        ni = ni.astype('float32')

        ni /= 255

        return ni
import tensorflow as tf

THRESHOLD = 0.05 

def f1(y_true, y_pred):

    # credits: https://www.kaggle.com/guglielmocamporese/macro-f1-score-keras

    #y_pred = K.round(y_pred)

    y_pred = K.cast(K.greater(K.clip(y_pred, 0, 1), THRESHOLD), K.floatx())

    tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)

    tn = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis=0)

    fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)

    fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)



    p = tp / (tp + fp + K.epsilon())

    r = tp / (tp + fn + K.epsilon())



    f1 = 2*p*r / (p+r+K.epsilon())

    f1 = tf.where(tf.math.is_nan(f1), tf.zeros_like(f1), f1)

    return K.mean(f1)



def create_model():

    model = Sequential()

    model.add(Conv2D(32, kernel_size=(3, 3),

                     activation='relu',

                     input_shape=input_shape))

    model.add(Conv2D(64, (3, 3), activation='relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, kernel_size=(3, 3),

                     activation='relu',

                     input_shape=input_shape))

    model.add(Conv2D(64, (3, 3), activation='relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, kernel_size=(3, 3),

                     activation='relu',

                     input_shape=input_shape))

    model.add(Conv2D(64, (3, 3), activation='relu'))

    model.add(Dropout(0.25))

    model.add(Flatten())

    model.add(Dense(128, activation='relu'))

    model.add(Dropout(0.5))

    model.add(Dense(num_classes, activation='sigmoid'))



    model.compile(loss=keras.losses.binary_crossentropy,

                  optimizer=keras.optimizers.Adam(lr=1e-3, decay=1e-3 / epochs),

                  metrics=['accuracy', f1])

    return model

model = create_model()



image_path = root_train_directory

training_generator = DataGenerator(train_img_names[:int(len(train_img_names)*0.8)], image_path, dim=SIZE,

                 n_channels=4)

validation_generator = DataGenerator(train_img_names[int(len(train_img_names)*0.8):], image_path, dim=SIZE,

                 n_channels=4)



H = model.fit_generator(generator=training_generator, validation_data=validation_generator,

          epochs=epochs,

          verbose=1,

         callbacks = [EarlyStopping(monitor='val_loss', patience=6, verbose=1), ModelCheckpoint(filepath='/tmp/weights.hdf5', verbose=1, save_best_only=True)])

model = create_model()

model.load_weights('/tmp/weights.hdf5')

score = model.evaluate(x_test, y_test, verbose=0)

print('Test loss:', score[0])

print('Test accuracy:', score[1])
model = create_model()

model.load_weights('/tmp/weights.hdf5')

score = model.evaluate(x_test, y_test, verbose=0)

print('Test loss:', score[0])

print('Test accuracy:', score[1])



# plot the training loss and accuracy

plt.style.use("ggplot")

plt.figure()

N = len(H.history['loss'])

plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")

plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")

plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")

plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")

plt.title("Training Loss and Accuracy")

plt.xlabel("Epoch #")

plt.ylabel("Loss/Accuracy")

plt.legend(loc="upper left")

plt.show()
def pred_to_textlabels(pred):

    classes = np.nonzero(pred[0])[0]

    text = ''

    for cl in classes:

        text+=str(name_label_dict[cl])+' '

    return text

    
# Let's do a random predictions

fig=plt.figure(figsize=(10, 16))



columns = 4

rows = 5

for i in range(1, columns*rows +1):

    num = randrange(len(train_img_names))

    num = randrange(1200)

    img_name = train_img_names[num]

    img = open_multilayer_image(img_name)

    img_to_predict = cv2.resize(img, SIZE)

    pred = model.predict(np.array([img_to_predict]))

    title = pred_to_textlabels(pred)

    

    real_label = ''

    for label in get_labels(img_name):

        real_label+=name_label_dict[label]+', '

    print('\'',real_label, '\'was predicted as\'',title,'\'' )

    sub = fig.add_subplot(rows, columns, i)

    # make title

    sub.set_title("\n".join(wrap(title,20)))

    plt.axis('off')

    plt.imshow(img[...,:-1])

plt.show();
test_df = pd.read_csv('/kaggle/input/human-protein-atlas-image-classification/sample_submission.csv')

imgs_test = test_df['Id'].to_list()

x_test = preprocess_x(1,imgs_test, test=True)



predictions = {}

for i in range(0, len(imgs_test)):

    sparse_vec = model.predict(np.array([x_test[i]]))

    predictions[imgs_test[i]]= np.nonzero(sparse_vec[0])[0]

predictions
test_df['Predicted'] = list(predictions.values())

test_df.to_csv('predictions.csv', index=False)