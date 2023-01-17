import os

import json



import numpy as np

import pandas as pd

import keras

from keras.callbacks import Callback

from keras.datasets import cifar10

from keras.preprocessing.image import ImageDataGenerator

from keras.models import Sequential

from keras.layers import Dense, Dropout, Activation, Flatten

from keras.layers import Conv2D, MaxPooling2D

from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

from sklearn.model_selection import train_test_split
os.listdir('../input')
# The data, split between train and test sets:

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

            num_classes=num_classes

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
batch_size = 64

num_classes = 14

epochs = 30

val_split = 0.1

save_dir = os.path.join(os.getcwd(), 'models')

model_name = 'keras_cnn_model.h5'
model = Sequential()

model.add(Conv2D(32, (3, 3), padding='same',

                 input_shape=x_train.shape[1:]))

model.add(Activation('relu'))

model.add(Conv2D(32, (3, 3)))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))



model.add(Conv2D(64, (3, 3), padding='same'))

model.add(Activation('relu'))

model.add(Conv2D(64, (3, 3)))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))



model.add(Flatten())

model.add(Dense(512))

model.add(Activation('relu'))

model.add(Dropout(0.5))

model.add(Dense(num_classes))

model.add(Activation('softmax'))
f1_metrics = Metrics()



model.compile(

    loss='categorical_crossentropy',

    optimizer='adam',

    metrics=['accuracy']

)



hist = model.fit(

    x_train, 

    y_train,

    batch_size=batch_size,

    epochs=epochs,

    callbacks=[f1_metrics],

    validation_split=val_split,

    shuffle=True

)
if not os.path.isdir(save_dir):

    os.makedirs(save_dir)

model_path = os.path.join(save_dir, model_name)

model.save(model_path)

print('Saved trained model at %s ' % model_path)
history_df = pd.DataFrame(hist.history)

history_df['val_f1'] = f1_metrics.val_f1s

history_df['val_precision'] = f1_metrics.val_precisions

history_df['val_recall'] = f1_metrics.val_recalls



history_df[['loss', 'val_loss']].plot()

history_df[['acc', 'val_acc']].plot()

history_df[['val_f1', 'val_precision', 'val_recall']].plot()
y_test = model.predict(x_test)



submission_df = pd.read_csv('../input/iwildcam-2019-fgvc6/sample_submission.csv')

submission_df['Predicted'] = y_test.argmax(axis=1)

print(submission_df.shape)

submission_df.head()
submission_df.to_csv('submission.csv',index=False)

history_df.to_csv('history.csv', index=False)



with open('history.json', 'w') as f:

    json.dump(hist.history, f)