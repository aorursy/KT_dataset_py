# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

from os import walk

print(os.listdir("../input"))

ppath = "../input/"



from PIL import Image

import pandas as pd



df_ = pd.read_csv(ppath+'train.csv')

print(df_.head(50))
train_array  = np.zeros((25000,32,32,3))

target_array = np.zeros((25000))

test_array   = np.zeros((25000,32,32,3))

# ------------------------------------------

idxIn=0

for (dirpath, dirnames, filenames) in walk(ppath+"train/train"):

    for filename in filenames:

        try:                  im = Image.open(dirpath+"/"+filename)

        except IOError as e:  continue

        x32_im = im.resize((32,32), Image.ANTIALIAS)

        

        train_array[idxIn,:,:,:] = np.asarray(x32_im)

        m_y = df_.loc[df_['id'] == filename].has_cactus

        target_array[idxIn]=m_y 

        # print(m_y)

        idxIn += 1

        #print(filename)

# ------------------------------------------

train_array = np.resize(train_array,(idxIn,32,32,3))

target_array = np.resize(target_array,(idxIn))

# ------------------------------------------

idxIn=0

for (dirpath, dirnames, filenames) in walk(ppath+"test/test"):

    for filename in filenames:

        try:                  im = Image.open(dirpath+"/"+filename)

        except IOError as e:  continue

        x32_im = im.resize((32,32), Image.ANTIALIAS)

        test_array[idxIn,:,:,:] = np.asarray(x32_im)

        idxIn += 1

        #print(filename)

# ------------------------------------------

test_array = np.resize(test_array,(idxIn,32,32,3))

# ------------------------------------------

import numpy as np

from keras.models import Sequential

from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D,BatchNormalization

from keras.layers import Activation, Dropout, Flatten



model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))

model.add(BatchNormalization())

model.add(Conv2D(32, (3, 3), activation='relu'))

model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2, 2)))

#model.add(Dropout(0.25))





model.add(Conv2D(64, (3, 3), activation='relu'))

model.add(BatchNormalization())

model.add(Conv2D(64, (3, 3), activation='relu'))

model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2, 2)))

# model.add(Dropout(0.25))



model.add(Flatten())

#model.add(Dense(600, activation='relu'))

# model.add(Dropout(0.5))



#model.add(Dense(64, activation='relu'))

#model.add(Dropout(0.2))

model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['accuracy'])

model.summary()



model.fit(train_array, target_array, epochs=20, batch_size=128, verbose=True)

# score = model.evaluate(x_test, y_test, batch_size=128)
df_s = pd.read_csv(ppath+'sample_submission.csv')

predicts = model.predict(test_array)



df_s['has_cactus'] = predicts

df_s.to_csv('sample_submission.csv', index=False)

predicts