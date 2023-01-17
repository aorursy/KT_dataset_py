import numpy as np

import cv2

import matplotlib.pyplot as plt

import sklearn

from collections import Counter

import glob

import pickle

import keras

from keras.preprocessing.image import ImageDataGenerator

from keras.models import Sequential

from keras.layers import Dense, Dropout, Activation, Flatten

from keras.layers import Conv2D, MaxPooling2D

from mpl_toolkits.axes_grid1 import AxesGrid

from sklearn.metrics import confusion_matrix

from keras.optimizers import SGD, Adam

from random import shuffle

%matplotlib inline
map_characters = {0: 'abraham_grampa_simpson', 1: 'apu_nahasapeemapetilon', 2: 'bart_simpson', 

        3: 'charles_montgomery_burns', 4: 'chief_wiggum', 5: 'comic_book_guy', 6: 'edna_krabappel', 

        7: 'homer_simpson', 8: 'kent_brockman', 9: 'krusty_the_clown', 10: 'lisa_simpson', 

        11: 'marge_simpson', 12: 'milhouse_van_houten', 13: 'moe_szyslak', 

        14: 'ned_flanders', 15: 'nelson_muntz', 16: 'principal_skinner', 17: 'sideshow_bob'}

pic_size = 64

batch_size = 32

epochs = 200

num_classes = len(map_characters)
def create_model_six_conv(input_shape):

    """

    CNN Keras model with 6 convolutions.

    :param input_shape: input shape, generally X_train.shape[1:]

    :return: Keras model, RMS prop optimizer

    """

    model = Sequential()

    model.add(Conv2D(32, (3, 3), padding='same', input_shape=input_shape))

    model.add(Activation('relu'))

    model.add(Conv2D(32, (3, 3)))

    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Dropout(0.2))



    model.add(Conv2D(64, (3, 3), padding='same'))

    model.add(Activation('relu'))

    model.add(Conv2D(64, (3, 3)))

    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Dropout(0.2))



    model.add(Conv2D(256, (3, 3), padding='same')) 

    model.add(Activation('relu'))

    model.add(Conv2D(256, (3, 3)))

    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Dropout(0.2))



    model.add(Flatten())

    model.add(Dense(1024))

    model.add(Activation('relu'))

    model.add(Dropout(0.5))

    model.add(Dense(num_classes, activation='softmax'))

    opt = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

    return model, opt



def load_model_from_checkpoint(weights_path, input_shape=(pic_size,pic_size,3)):

    model, opt = create_model_six_conv(input_shape)

    model.load_weights(weights_path)

    model.compile(loss='categorical_crossentropy',

              optimizer=opt,

              metrics=['accuracy'])

    return model

  

def load_test_set(path):

    pics, labels = [], []

    reverse_dict = {v:k for k,v in map_characters.items()}

    for pic in glob.glob(path+'*.*'):

        char_name = "_".join(pic.split('/')[4].split('_')[:-1])

        if char_name in reverse_dict:

            temp = cv2.imread(pic)

            temp = cv2.resize(temp, (pic_size,pic_size)).astype('float32') / 255.

            pics.append(temp)

            labels.append(reverse_dict[char_name])

    X_test = np.array(pics)

    y_test = np.array(labels)

    y_test = keras.utils.to_categorical(y_test, num_classes)

    print("Test set", X_test.shape, y_test.shape)

    return X_test, y_test
model = load_model_from_checkpoint('../input/the-simpsons-characters-dataset/weights.best.hdf5')

X_test, y_test = load_test_set("../input/the-simpsons-characters-dataset/kaggle_simpson_testset/")

y_pred = model.predict(X_test)
print('\n', sklearn.metrics.classification_report(np.where(y_test > 0)[1], 

                                                  np.argmax(y_pred, axis=1), 

                                                  target_names=list(map_characters.values())), sep='')
plt.figure(figsize=(8,8))

cnf_matrix = sklearn.metrics.confusion_matrix(np.where(y_test > 0)[1], np.argmax(y_pred, axis=1))

classes = list(map_characters.values())

plt.imshow(cnf_matrix, interpolation='nearest')

plt.colorbar()

tick_marks = np.arange(len(classes))

_ = plt.xticks(tick_marks, classes, rotation=90)

_ = plt.yticks(tick_marks, classes)
F = plt.figure(1, (15,20))

grid = AxesGrid(F, 111, nrows_ncols=(4, 4), axes_pad=0, label_mode="1")



for i in range(16):

    char = map_characters[i]

    image = cv2.imread(np.random.choice([k for k in glob.glob('../input/the-simpsons-characters-dataset/kaggle_simpson_testset/*.*') if char in k]))

    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    pic = cv2.resize(image, (64, 64)).astype('float32') / 255.

    a = model.predict(pic.reshape(1, 64, 64,3))[0]

    actual = char.split('_')[0].title()

    text = sorted(['{:s} : {:.1f}%'.format(map_characters[k].split('_')[0].title(), 100*v) for k,v in enumerate(a)], 

       key=lambda x:float(x.split(':')[1].split('%')[0]), reverse=True)[:3]

    img = cv2.resize(img, (352, 352))

    cv2.rectangle(img, (0,260),(215,352),(255,255,255), -1)

    font = cv2.FONT_HERSHEY_SIMPLEX

    cv2.putText(img, 'Actual : %s' % actual, (10, 280), font, 0.7,(0,0,0),2,cv2.LINE_AA)

    for k, t in enumerate(text):

        cv2.putText(img, t,(10, 300+k*18), font, 0.65,(0,0,0),2,cv2.LINE_AA)

    grid[i].imshow(img)