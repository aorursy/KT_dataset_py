# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import gc



import keras as k

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten

from keras.layers import Conv2D, MaxPooling2D



import cv2

from tqdm import tqdm



from multiprocessing import Pool, cpu_count



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
x_train0 = []

x_test0 = []

y_train0 = []



df_train = pd.read_csv('../input/train_v2.csv')



labels = df_train['tags'].str.get_dummies(sep=' ').columns



label_map = {l: i for i, l in enumerate(labels)}

inv_label_map = {i: l for l, i in label_map.items()}



for f, tags in tqdm(df_train.values, miniters=1000):

    img = cv2.imread('../input/train-jpg/{}.jpg'.format(f))

    targets = np.zeros(17)

    for t in tags.split(' '):

        targets[label_map[t]] = 1 

    x_train0.append(cv2.resize(img, (32, 32)))

    y_train0.append(targets)

    

y_train0 = np.array(y_train0, np.uint8)

x_train0 = np.array(x_train0, np.float16) / 255.



print(x_train0.shape)

print(y_train0.shape)
df_test = pd.read_csv('../input/sample_submission_v2.csv')



for f, tags in tqdm(df_test.values, miniters=1000):

    img = cv2.imread('../input/test-jpg-v2/{}.jpg'.format(f))

    targets = np.zeros(17)

    for t in tags.split(' '):

        targets[label_map[t]] = 1 

    x_test0.append(cv2.resize(img, (32, 32)))

    

x_test0 = np.array(x_test0, np.float16) / 255.
split = 35000

x_train, x_valid, y_train, y_valid = x_train0[:split], x_train0[split:], y_train0[:split], y_train0[split:]



model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3),

                 activation='relu',

                 input_shape=(32, 32, 3)))



model.add(Conv2D(64, (3, 3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(128, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(17, activation='sigmoid'))



model.compile(loss='binary_crossentropy', # We NEED binary here, since categorical_crossentropy l1 norms the output before calculating loss.

              optimizer='adam',

              metrics=['accuracy'])

              

model.fit(x_train, y_train,

          batch_size=128,

          epochs=3,

          verbose=1,

          validation_data=(x_valid, y_valid))

          

from sklearn.metrics import fbeta_score



p_train = model.predict(x_train0, batch_size=128,verbose=2)

p_test = model.predict(x_test0, batch_size=128,verbose=2)

# print(fbeta_score(y_valid, np.array(p_valid) > 0.2, beta=2, average='samples'))
# Saving predicted probability and ground truth for Train Dataset

# Compute the best threshold externally

print(labels)

chk_output = pd.DataFrame()

for index in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]:

    chk_output['class %d' % index] = p_train[:,index-1]

chk_output.to_csv('predicted_probability.csv', index=False)

for index in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]:

    chk_output['class %d' % index] = y_train0[:,index-1]

chk_output.to_csv('true_label.csv', index=False)
values_test = (p_test > .222222)*1.0        # before multiplying by 1.0, this appears as an array of True and False

values_test = np.array(values_test, np.uint8)



print(values_test)

# Build Submission, using label outputted from long time ago

test_labels = []

for row in range(values_test.shape[0]):

    test_labels.append(' '.join(labels[values_test[row,:]==1]))

Submission_PDFModel = df_test.copy()

Submission_PDFModel.drop('tags', axis = 1)

Submission_PDFModel['tags'] = test_labels

Submission_PDFModel.to_csv('sub_simple_keras_32x32_online.csv', index = False)