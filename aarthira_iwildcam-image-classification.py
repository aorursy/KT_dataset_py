# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import math

import matplotlib.pyplot as plt

import cv2

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from PIL import Image

import os,sys

print(os.listdir("../input"))





# Any results you write to the current directory are saved as output.
label_df = pd.read_csv('../input/iwildcam-2019-fgvc6/train.csv')

submission_df = pd.read_csv('../input/iwildcam-2019-fgvc6/sample_submission.csv')

label_df.head()



def samples(df, columns=3, rows=3):

    fig=plt.figure(figsize=(6*columns, 3*rows))



    for i in range(columns*rows):

        img_path = df.loc[i,'file_name']

        img_id = df.loc[i,'category_id']

        img = cv2.imread(f'../input/train_images/{img_path}')

        fig.add_subplot(rows, columns, i+1)

        plt.title(img_id)

        plt.imshow(img)



samples(label_df)

def pad_width(im, new_shape, is_rgb=True):

    pad_diff = new_shape - im.shape[0], new_shape - im.shape[1]

    t, b = math.floor(pad_diff[0]/2), math.ceil(pad_diff[0]/2)

    l, r = math.floor(pad_diff[1]/2), math.ceil(pad_diff[1]/2)

    if is_rgb:

        width = ((t,b), (l,r), (0, 0))

    else:

        width = ((t,b), (l,r))

    return pad_width



def pad_and_resize(img_path, dataset, pad=False, desired_size=32):

    img = cv2.imread(f'../input/{dataset}_images/{img_path}.jpg')

    

    if pad:

        width = pad_width(img, max(img.shape))

        padded = np.pad(img, width=width, mode='constant', constant_values=0)

    else:

        padded = img

    

    resized = cv2.resize(padded, (desired_size,)*2).astype('uint8')

    

    return resized

%%time

train_resized = []

test_resized = []



for image_id in label_df['id']:

    train_resized.append(

        pad_and_resize(image_id, 'train')

    )



for image_id in submission_df['Id']:

    test_resized.append(

        pad_and_resize(image_id, 'test')

    )

X_train = np.stack(train_resized)

X_test = np.stack(test_resized)



target_dummies = pd.get_dummies(label_df['category_id'])

train_label = target_dummies.columns.values

y_train = target_dummies.values



print(X_train.shape)

print(X_test.shape)

print(y_train.shape)

np.save('Resized_Xtrain.npy',X_train)

np.save('Resized_ytrain.npy',y_train)

np.save('Resized_Xtest.npy',X_test)

print(os.listdir('../input/reduceddata/wildcam-reduced'))
from zipfile import ZipFile

#zf = ZipFile('../input/reduceddata/wildcam-reduced.zip','r')

#zf.extractall('..input/')

#zf.close()

y_train = np.load('../input/reduceddata/wildcam-reduced/y_train.npy')

X_train = np.load('../input/reduceddata/wildcam-reduced/X_train.npy')

X_test = np.load('../input/reduceddata/wildcam-reduced/X_test.npy')
print('X_train shape is',X_train.shape)

print('X_test shape is', X_test.shape)

print('y_train.shape is',y_train.shape)
X_train = X_train.astype('float32')

X_test = X_test.astype('float32')

X_train /= 255

X_test /= 255
from keras.applications import DenseNet121

from keras.layers import *

from keras.models import Sequential

dense_network = DenseNet121(input_shape = (32, 32, 3),include_top = False, classes = 1000)

model = Sequential()

model.add(dense_network)

model.add(GlobalAveragePooling2D())

model.add(Dropout(0.5))



model.add(Dense(14, activation='softmax'))



model.summary()

model.compile(loss='categorical_crossentropy',

              optimizer='adam',

              metrics=['accuracy'])



from keras.callbacks import Callback

from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

class Metrics(Callback):

    def on_train_begin(self, logs={}):

        self.val_f1s = []

        self.val_recalls = []

        self.val_precisions = []



    def on_epoch_end(self, epoch, logs={}):

        X_val, y_val = self.validation_data[:2]

        y_pred = self.model.predict(X_val)



        y_pred_cat = keras.utils.to_categorical(

            y_pred.argmax(axis=1),

            num_classes=14

        )

        _val_f1 = f1_score(y_val, y_pred_cat, average='macro')

        _val_recall = recall_score(y_val, y_pred_cat, average='macro')

        _val_precision = precision_score(y_val, y_pred_cat, average='macro')



        self.val_f1s.append(_val_f1)

        self.val_recalls.append(_val_recall)

        self.val_precisions.append(_val_precision)



        print((f"val_f1: {_val_f1:.4f}"

               f" — val_precision: {_val_precision:.4f}"

               f" — val_recall: {_val_recall:.4f}"))



        return



f1_metrics = Metrics()

import keras

from keras.callbacks import  ModelCheckpoint



checkpoint = ModelCheckpoint(

    'model.h5', 

    monitor='val_acc', 

    verbose=1, 

    save_best_only=True, 

    save_weights_only=False,

    mode='auto'

)

history = model.fit(

    x=X_train,

    y=y_train,

    batch_size=64,

    epochs=10,

    callbacks=[f1_metrics],

    validation_split=0.2

)



fig = plt.subplots(figsize=(8,8))

plt.plot(history.history['loss'],color='g')

plt.plot(history.history['val_loss'],color='r')

plt.legend(['training','validation'])

plt.show()

fig = plt.subplots(figsize=(8,8))

plt.plot(history.history['acc'],color='g')

plt.plot(history.history['val_acc'],color='r')

plt.legend(['training','validation'])

plt.show()
result = model.predict(X_test)

submission_df['Predicted'] = result.argmax(axis=1)

submission_df.head()
submission_df.to_csv('final.csv',index=False)