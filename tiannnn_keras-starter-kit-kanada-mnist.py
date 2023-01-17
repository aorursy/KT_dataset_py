# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

from keras import backend as K

import keras

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten

from keras.layers import Conv2D, MaxPooling2D

from keras.callbacks import ModelCheckpoint
path = '../input/Kannada-MNIST/'

file_name_dict= {'dig':'Dig-MNIST','test':'test','train':'train','sample':'sample_submission'}

df_dict = {key:pd.read_csv(path + value + '.csv') for key, value in file_name_dict.items()}
def get_xy(df):

    X = df.drop('label', axis=1)

    y = df[['label']]

    X = X_reshaper(X)

    y = label_reshaper(y)

    return X, y



def X_reshaper(df):

    X = df.copy().values

    X = X.reshape(X.shape[0], n_rows, n_columns, 1)

    X = X.astype('float32')

    return X



def label_reshaper(y, num_classes=10):

    y = keras.utils.to_categorical(y, num_classes)

    return y



def recall_m(y_true, y_pred):

        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))

        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))

        recall = true_positives / (possible_positives + K.epsilon())

        return recall



def precision_m(y_true, y_pred):

        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))

        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))

        precision = true_positives / (predicted_positives + K.epsilon())

        return precision

    

def create_model(input_shape, num_classes):

    model = Sequential()

    model.add(Conv2D(32, kernel_size=(3, 3),

                     activation='relu',

                     input_shape=input_shape))

    model.add(Conv2D(64, (3, 3), activation='relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Dropout(0.25))

    model.add(Conv2D(128, (3, 3), activation='relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Dropout(0.25))

    model.add(Conv2D(256, (3, 3), activation='relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Dropout(0.25))

    model.add(Flatten())

    model.add(Dense(256, activation='relu'))

    model.add(Dropout(0.5))

    model.add(Dense(num_classes, activation='softmax'))



    model.compile(loss=keras.losses.categorical_crossentropy,

                  optimizer=keras.optimizers.Adam(),

                  metrics=['accuracy', recall_m, precision_m])

    return model



n_rows, n_columns = 28,28

batch_size = 128

num_classes = 10

epochs = 100

input_shape = (n_rows, n_columns, 1)

num_classes = 10

X_train, y_train = get_xy(df_dict['train'])

X_val, y_val = get_xy(df_dict['dig'])





checkpoint = ModelCheckpoint('/kaggle/working/weights.hdf5', monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')

callbacks_list = [checkpoint]

model = create_model(input_shape, num_classes)

model.fit(X_train, y_train,

          batch_size=batch_size,

          epochs=epochs,

          verbose=1,

          validation_data=(X_val, y_val),

          callbacks=callbacks_list)

score = model.evaluate(X_val, y_val, verbose=0)
X_test_df  = df_dict['test'].copy().drop('id', axis=1)

X_test = X_reshaper(X_test_df)

model = create_model(input_shape, num_classes)

model.load_weights('/kaggle/working/weights.hdf5')

y_pred = model.predict(X_test)

y_pred_classes = np.argmax(y_pred, axis = 1)

output = pd.DataFrame({'id': df_dict['test']['id'],

                       'label': y_pred_classes})



output.to_csv('submission.csv', index=False)