# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
path = '../input/SampleMoviePosters'
import glob
import scipy.misc
image_glob = glob.glob(path+"/"+"*.jpg")
img_dict = {}
def get_id(filename):
    index_s = filename.rfind("/")+1
    index_f = filename.rfind(".jpg")
    return filename[index_s:index_f]
#Populate image dict
_ = [img_dict.update({get_id(fn):scipy.misc.imread(fn)}) for fn in image_glob]

df = pd.read_csv("../input/MovieGenre.csv",encoding="ISO-8859-1")
genres = []
length = len(df)
for n in range(len(df)):
    g = str(df.loc[n]["Genre"])
    genres.append(g)
    
classes = list(set(genres))
classes.sort()
num_classes = len(classes)

def get_classes_from_movie(movie_id):
    y = np.zeros(num_classes)
    g = str(df[df['imdbId']==movie_id]['Genre'].values[0])
    y[classes.index(g)] = 1
    return y    
import random

"""
Some relatively simple image preprocessing
"""
def preprocess(img,size=32):
    img = scipy.misc.imresize(img,(size,size))
    img = img.astype(np.float32)
    img = (img / 127.5) - 1.
    return img
    

def get_dataset(train_size,img_size=32):
        #indices = np.random.randint(0,len(list(img_dict.keys()))-1,batch_size)
        indices = random.sample(range(len(list(img_dict.keys()))),train_size)
        x = []
        y = []
        x_test = []
        y_test = []
        for i in range(len(list(img_dict.keys()))):
            id_key = int(list(img_dict.keys())[i])
            if i in indices:
                x.append(preprocess(img_dict[list(img_dict.keys())[i]],size=img_size))
                y.append(get_classes_from_movie(id_key))
            else:
                x_test.append(preprocess(img_dict[list(img_dict.keys())[i]],size=img_size))
                y_test.append(get_classes_from_movie(id_key))
        return x,y,x_test,y_test
#Constant to keep track of our image size
SIZE = 128   
x,y,x_test,y_test = get_dataset(900,img_size=SIZE)
x = np.asarray(x)
y = np.asarray(y)
x_test = np.asarray(x_test)
y_test = np.asarray(y_test)

#import tensorflow as tf
#sess = tf.Session()
from keras import backend as K
#K.set_session(sess)
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D

model = Sequential()
model.add(Conv2D(32,kernel_size=(3,3),activation='relu',input_shape=(SIZE,SIZE,3)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])


model.fit(x, y,
          batch_size=50,
          epochs=5,
          verbose=1,
          validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
pred = model.predict(np.asarray([x_test[5]]))
print(pred)
print(np.argmax(pred))
print(np.argmax(y_test[5]))