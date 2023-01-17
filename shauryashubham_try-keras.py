# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

%matplotlib inline

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train_v2.csv')

test = pd.read_csv('../input/sample_submission_v2.csv')
labels = train['tags'].apply(lambda x: x.split(' '))
label_list=[]

for i in labels:

    for j in i:

        if j not in label_list:

            label_list.append(j)
x_train = []

x_test = []

y_train = []
label_map = {l: i for i, l in enumerate(label_list)}
label_map
from tqdm import tqdm

import cv2
for f, tags in tqdm(train.values, miniters=1000):

    img = cv2.imread('../input/train-jpg/{}.jpg'.format(f))

    img = cv2.resize(img,(64,64))

    targets = np.zeros(17)

    for t in tags.split(' '):

        targets[label_map[t]] = 1

    x_train.append(img)

    y_train.append(targets)
for f, tags in tqdm(test.values, miniters=1000):

    img = cv2.imread('../input/test-jpg-v2/{}.jpg'.format(f))

    img = cv2.resize(img,(64,64))

    x_test.append(cv2.resize(img,(64, 64)))
y_train = np.array(y_train)

x_train = np.array(x_train, np.float32)/255.0

x_test  = np.array(x_test, np.float32)/255.0
print(x_train.shape)

print(x_test.shape)

print(y_train.shape)
import keras

from keras.models import Sequential

from keras.layers.core import Dense, Dropout, Activation, Flatten

from keras.layers.convolutional import Conv2D, MaxPooling2D

from keras.optimizers import Adam

np.random.seed(1671)
model = Sequential()

model.add(Conv2D(32, (3,3), strides=(1,1), padding='same', input_shape=(64,64,3)))

model.add(Activation('relu'))

model.add(Conv2D(32, (3,3),  padding='same' ,strides=(1,1)))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

model.add(Conv2D(64, (3,3),  padding='same' ,strides=(1,1)))

model.add(Activation('relu'))

model.add(Conv2D(64, (3,3), strides=(1,1)))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2,2), padding='same', strides=(2,2)))

model.add(Conv2D(128, (3,3),  padding='same' ,strides=(1,1)))

model.add(Activation('relu'))

model.add(Conv2D(128, (3,3),  padding='same' ,strides=(1,1)))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2,2), padding='same', strides=(2,2)))

model.add(Flatten())

model.add(Dense(128))

model.add(Activation('relu'))

model.add(Dropout(0.25))

model.add(Dense(17))

model.add(Activation('softmax'))

model.summary()
OPTIMIZER = Adam( lr=0.001)

model.compile(loss='binary_crossentropy', optimizer=OPTIMIZER , metrics=['accuracy'])

history = model.fit(x_train, y_train, batch_size=64, epochs=2, verbose=1, validation_split=0.25)
OPTIMIZER.lr = 0.0001

history = model.fit(x_train, y_train, batch_size=64, epochs=2, verbose=1, validation_split=0.25)
y_test=[]

p_test = model.predict(x_test)

y_test.append(p_test)
p_test.shape
p_test
y_test[0].shape
result = pd.DataFrame(p_test, columns = label_list)

result
from tqdm import tqdm

preds = []

for i in tqdm(range(result.shape[0]), miniters=1000):

    a = result.ix[[i]]

    a = a.apply(lambda x: x > 0.2, axis=1)

    a = a.transpose()

    a = a.loc[a[i] == True]

    ' '.join(list(a.index))

    preds.append(' '.join(list(a.index)))
preds
test['tags'] = preds

test.to_csv('submission.csv', index=False)
test.to_csv('sample_submission_v2.csv', index=False)
test.to_csv('sample_submission_v2.csv', index=False)
test.to_csv('resul.csv', index=False)
k = pd.read_csv('sample_submission_v2.csv')

k.head()
print(check_output(["ls", "./"]).decode("utf8"))
ls
pwd