# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import os

import json



import numpy as np

import pandas as pd

import keras

from keras import layers

from keras.applications import DenseNet121

from keras.callbacks import Callback, ModelCheckpoint

from keras.preprocessing.image import ImageDataGenerator

from keras.layers import Dense, Dropout, Activation, Flatten

from keras.layers import Conv2D, MaxPooling2D

from keras.models import Sequential

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
x_train = np.load('../input/reducing-image-sizes-to-32x32/X_train.npy')

x_test = np.load('../input/reducing-image-sizes-to-32x32/X_test.npy')

y_train = np.load('../input/reducing-image-sizes-to-32x32/y_train.npy')



print('x_train shape:', x_train.shape)

print(x_train.shape[0], 'train samples')

print(x_test.shape[0], 'test samples')
# Convert the images to float and scale it to a range of 0 to 1

x_train = x_train.astype('float32')

x_test = x_test.astype('float32')

x_train /= 255.

x_test /= 255.
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
densenet = DenseNet121(

    weights='../input/densenet-keras/DenseNet-BC-121-32-no-top.h5',

    include_top=False,

    input_shape=(32,32,3)

)
model = Sequential()

model.add(densenet)

model.add(layers.GlobalAveragePooling2D())

model.add(layers.Dense(14, activation='softmax'))
model.compile(loss='categorical_crossentropy',

              optimizer='adamax',

              metrics=['accuracy'])



model.summary()
checkpoint = ModelCheckpoint(

    'model.h5', 

    monitor='val_acc', 

    verbose=1, 

    save_best_only=True, 

    save_weights_only=False,

    mode='auto'

)



history = model.fit(

    x=x_train,

    y=y_train,

    batch_size=64,

    epochs=7,

    callbacks=[checkpoint, f1_metrics],

    validation_split=0.1

)
with open('history.json', 'w') as f:

    json.dump(history.history, f)



history_df = pd.DataFrame(history.history)

history_df['val_f1'] = f1_metrics.val_f1s

history_df['val_precision'] = f1_metrics.val_precisions

history_df['val_recall'] = f1_metrics.val_recalls

history_df[['loss', 'val_loss']].plot()

history_df[['acc', 'val_acc']].plot()

history_df[['val_f1', 'val_precision', 'val_recall']].plot()
model.load_weights('model.h5')

y_test = model.predict(x_test)



submission_df = pd.read_csv('../input/iwildcam-2019-fgvc6/sample_submission.csv')

submission_df['Predicted'] = y_test.argmax(axis=1)



submission_df['Species'] = 'default'

species = []

for index, row in submission_df.iterrows():

	if row['Predicted'] == 0:

		species.append('empty')

	elif row['Predicted'] == 1:

		species.append('deer')

	elif row['Predicted'] == 2:

		species.append('moose')

	elif row['Predicted'] == 3:

		species.append('squirrel')

	elif row['Predicted'] == 4:

		species.append('rodent')

	elif row['Predicted'] == 5:

		species.append('small_mammal')

	elif row['Predicted'] == 6:

		species.append('elk')

	elif row['Predicted'] == 7:

		species.append('pronghorn_antelope')

	elif row['Predicted'] == 8:

		species.append('rabbit')

	elif row['Predicted'] == 9:

		species.append('bighorn_sheep')

	elif row['Predicted'] == 10:

		species.append('fox')

	elif row['Predicted'] == 11:

		species.append('coyote')

	elif row['Predicted'] == 12:

		species.append('black_bear')

	elif row['Predicted'] == 13:

		species.append('raccoon')

	elif row['Predicted'] == 14:

		species.append('skunk')

	elif row['Predicted'] == 15:

		species.append('wolf')

	elif row['Predicted'] == 16:

		species.append('bobcat')

	elif row['Predicted'] == 17:

		species.append('cat')

	elif row['Predicted'] == 18:

		species.append('dog')

	elif row['Predicted'] == 19:

		species.append('oppossum')

	elif row['Predicted'] == 20:

		species.append('bison')

	elif row['Predicted'] == 21:

		species.append('mountain_goat')

	elif row['Predicted'] == 22:

		species.append('mountain_lion')

        

submission_df['Species'] = species

        

print(submission_df.shape)

submission_df.head()
submission_df.to_csv('submission.csv',index=False)