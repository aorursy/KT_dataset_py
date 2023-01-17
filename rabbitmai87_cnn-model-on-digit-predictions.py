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



import matplotlib.pyplot as plt

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Conv1D, MaxPool1D, Conv2D, MaxPool2D

from keras.optimizers import Adam

from sklearn.model_selection import train_test_split

from keras.utils.np_utils import to_categorical

from sklearn.metrics import confusion_matrix

from keras.callbacks import ReduceLROnPlateau

from keras.applications.vgg16 import VGG16
base_dir = os.path.join('..', 'input')
#explore sample_submission file

submission = pd.read_csv(os.path.join(base_dir,'sample_submission.csv'))

submission.head()
submission['Label'].unique()
len(submission)
#load train & test sets

train = pd.read_csv(os.path.join(base_dir,'train.csv'))

test = pd.read_csv(os.path.join(base_dir, 'test.csv'))

train.head()
test.head()
print("Training set has {} images".format(len(train)))

print("Training set has {} columns".format(len(train.columns)))

print("\nTest set has {} images".format(len(test)))

print("Test set has {} columns".format(len(test.columns)))
train['label'].value_counts().plot(kind='bar')
features = train.drop(columns='label')

labels = train['label']
#convert features from dataframe to numpy array

features = features.values
features.shape
#convert test from dataframe to numpy array

test = test.values

test.shape
num_rows = 5

num_cols = 5



fig, ax = plt.subplots(num_rows, num_cols, figsize=(13,14))



i = 0



for each_row in ax:

    

    for each_row_col in each_row:

        each_row_col.imshow(features[i].reshape(28,28))

        each_row_col.set_title('True label: '+str(labels[i]))

        each_row_col.axis('off')

        i += 1
#reason of data normalization is for faster training

#after normalization, each pixel is compressed to between 0 & 1

features = features / 255

test = test / 255
x_train, x_valid, y_train, y_valid = train_test_split(features, labels, test_size=0.2, random_state = 1234)
print("x_train size: ", len(x_train))

print("y_train size: ", len(x_valid))
num_classes = len(y_train.unique())



y_train = to_categorical(y_train, num_classes=num_classes)

y_valid = to_categorical(y_valid, num_classes=num_classes)
print("x_train's shape: ", x_train.shape)

print("x_valid's shape: ", x_valid.shape)



print("\ny_train's shape now: ", y_train.shape)

print("y_valid 's shape now: ", y_valid.shape)
model = Sequential([Dense(256, activation='relu', input_shape=[784]),

                    Dropout(0.2),

                    Dense(128, activation='relu'),

                    Dropout(0.4),

                    Dense(num_classes, activation='softmax') 

                   ])
model.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy', metrics=['acc'])
history = model.fit(x_train, y_train, epochs=10, validation_data=(x_valid,y_valid))
_, acc_valid = model.evaluate(x_valid,y_valid)

print("Validation accuracy: ", acc_valid)
history.history.keys()
def visual(model_history):

    

    fig, ax = plt.subplots(1,2,figsize=(15,6))

    

    acc = model_history.history['acc']

    val_acc = model_history.history['val_acc']

    

    ax[0].plot(range(1, len(model_history.history['acc'])+1), acc, label='traing accuracy')

    ax[0].plot(range(1, len(model_history.history['acc'])+1), val_acc, label='validation accuracy')

    ax[0].set_title('Model Accuracy')

    ax[0].set_xlabel('epochs')

    ax[0].set_ylabel('accuracy')

    ax[0].legend()

    

    loss = model_history.history['loss']

    val_loss = model_history.history['val_loss']

    

    ax[1].plot(range(1, len(model_history.history['acc'])+1), loss, label="train loss")

    ax[1].plot(range(1, len(model_history.history['acc'])+1), val_loss, label='validation loss')

    ax[1].set_title('Model Loss')

    ax[1].set_xlabel('epochs')

    ax[1].set_ylabel('loss')

    ax[1].legend()
visual(history)
y_pred = model.predict(x_valid)



y_pred_classes = np.argmax(y_pred, axis=1)



y_true_classes = np.argmax(y_valid, axis =1)
confusion_mtx = confusion_matrix(y_true_classes, y_pred_classes)
confusion_mtx
label_frac_error = 1 - np.diag(confusion_mtx) / np.sum(confusion_mtx, axis = 1)

plt.bar(np.arange(10), label_frac_error)

plt.xlabel('True Label')

plt.ylabel('Fraction classified incorrectly')
#reshape x_train and x_valid to shape = (28,28)

cnn_xtrain = x_train.reshape(x_train.shape[0], 28,28,1)

cnn_xvalid = x_valid.reshape(x_valid.shape[0], 28,28,1)

cnn_test = test.reshape(test.shape[0], 28,28,1)
print(cnn_xtrain.shape)

print(cnn_xvalid.shape)

print(cnn_test.shape)
cnn = Sequential([Conv2D(36, kernel_size=(2,2), activation='relu', padding='same', input_shape=(28,28,1)),

                 MaxPool2D(pool_size=(2,2)),

                 Dropout(0.2),

                  

                 Conv2D(64, kernel_size=(2,2), activation='relu', padding='same'),

                 MaxPool2D(pool_size=(2,2)),

                 Dropout(0.4),

                  

                 Conv2D(128, kernel_size=(2,2), activation='relu', padding='same'),

                 MaxPool2D(pool_size=(2,2)),

                 Dropout(0.4),

                  

                 Flatten(),

                 Dense(256, activation='relu'),

                 Dropout(0.5),

                 Dense(128,activation='relu'),

                 Dropout(0.5),

                 Dense(10, activation='softmax')

                 ])
cnn.summary()
cnn.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy', metrics=['acc'])
lr_reduction = ReduceLROnPlateau(monitor='val_acc',

                                 patience = 3,

                                 verbose = 1,

                                 factor = 0.5,

                                 min_lr = 0.000001

                                )
cnn_history = cnn.fit(cnn_xtrain, y_train, epochs=40, batch_size=30, validation_data=(cnn_xvalid,y_valid),callbacks=[lr_reduction])
visual(cnn_history)
test_pred = cnn.predict(cnn_test)
test_pred_classes = np.argmax(test_pred, axis =1)
test_pred_classes
plt.imshow(cnn_test[3].reshape(28,28))
submission['Label'] = test_pred_classes
submission.head()
submission.to_csv('digits_classification_sub.csv',index=False)