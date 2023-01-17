# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from tqdm import tqdm

import matplotlib.pyplot as plt

import os

%matplotlib inline

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv')
train.head()
Labels = train['label']

train = train.drop('label', axis=1)
#Converting our Dataframe to a matrix 

train_mat = train.as_matrix()
Images = []

for i in tqdm(range(train_mat.shape[0])):

    im = train_mat[i]

    im = im.reshape((28,28))

    Images.append(im)
Images = np.array(Images)
Images.shape
plt.imshow(Images[3], cmap='gray')

print(Labels[3])
Images = Images/255.
import seaborn as sns

sns.countplot(Labels)
from sklearn.model_selection import train_test_split



xtr, xv, ytr, yv = train_test_split(Images, Labels, random_state = 45, test_size=0.2)
xtr = xtr.reshape(xtr.shape[0],28,28,1)

xv = xv.reshape(xv.shape[0],28,28,1)
from keras.preprocessing.image import ImageDataGenerator



train_gen = ImageDataGenerator(featurewise_center=True,

    featurewise_std_normalization=True,

    rotation_range=20,

    width_shift_range=0.2,

    height_shift_range=0.2, zoom_range=0.2)



train_gen.fit(xtr)
from keras.utils import to_categorical



ytr1 = to_categorical(ytr)

yv1 = to_categorical(yv)



yv1[:5]
from keras.layers import Conv2D, MaxPooling2D, Dropout, BatchNormalization, Flatten, Dense, Activation

from keras import optimizers

from keras.models import Sequential
model = Sequential()

model.add(Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=(28,28,1)))

model.add(Conv2D(32, kernel_size=(3,3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))



model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))

model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))



model.add(Flatten()) # Flattening the 2D arrays for fully connected layers

model.add(Dense(256, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(10,activation='softmax'))
model.summary()
model.compile(optimizer=optimizers.Adam(lr=0.0001),

              loss='categorical_crossentropy', 

              metrics=['accuracy'])
from keras.callbacks import EarlyStopping, ModelCheckpoint



es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)

mc = ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)
history = model.fit(xtr, ytr1, validation_data=(xv, yv1)

                              , epochs=20

                              , batch_size=32

                             , callbacks=[es, mc])
from keras.models import load_model



final_model = load_model('best_model.h5')
test = pd.read_csv('../input/test.csv')
test_mat = test.as_matrix()
Images_test = []

for i in tqdm(range(test_mat.shape[0])):

    im = test_mat[i]

    im = im.reshape((28,28))

    Images_test.append(im)
Images_test = np.array(Images_test)
Images_test = Images_test.reshape(Images_test.shape[0],28,28,1)
predictions = final_model.predict(Images_test)
predictions
pred_labels = []



for i in predictions:

    l = np.argmax(i)

    pred_labels.append(l)
#plot of the predicted labels 

sns.countplot(pred_labels)
submission = pd.read_csv('../input/sample_submission.csv')
imageid = submission['ImageId']
data_final = {'ImageId': imageid, 'Label': pred_labels }
final_sub = pd.DataFrame(data=data_final)

final_sub.head()
#Exporting to CSV for submission

final_sub.to_csv('final_submission.csv', sep=',', index=False)