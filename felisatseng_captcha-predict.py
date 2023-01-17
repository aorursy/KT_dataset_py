# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from PIL import Image, ImageDraw, ImageFont

import numpy as np

import pandas as pd

from skimage.util import invert



print("Reading training data...")

images = [invert(np.array(Image.open("/kaggle/input/captcha-images/images/images/" + str(index) + ".jpg")))/255.0 for index in range(3000)]

train_data = np.stack(images)



traincsv = pd.read_csv("/kaggle/input/captcha-images/label_file.csv", index_col=None)

label_data = traincsv.values.reshape(3000, 5, 29)

train_label = np.hsplit(label_data, 5)

for i in range(len(train_label)):

    train_label[i] = train_label[i].reshape(3000, 29)



print("Shape of train data:", train_data.shape)
from keras.models import Sequential,load_model,Model

from keras.layers import Input, Dense, Dropout, Flatten, Conv2D, MaxPooling2D

from keras.layers.normalization import BatchNormalization

from keras.utils  import np_utils

from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard



#creat CNN model

print('Creating CNN model...')

tensor_in = Input((train_data.shape[1], train_data.shape[2], 3))

tensor_out = tensor_in



#tensor_out = Conv2D(filters=16, kernel_size=(3, 3), padding='same', activation='relu')(tensor_out)

#tensor_out = Conv2D(filters=16, kernel_size=(3, 3), activation='relu')(tensor_out)

#tensor_out = MaxPooling2D(pool_size=(2, 2))(tensor_out)



tensor_out = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(tensor_out)

tensor_out = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(tensor_out)

tensor_out = MaxPooling2D(pool_size=(2, 2))(tensor_out)



tensor_out = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(tensor_out)

tensor_out = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(tensor_out)

tensor_out = MaxPooling2D(pool_size=(2, 2))(tensor_out)



tensor_out = Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu')(tensor_out)

tensor_out = Conv2D(filters=128, kernel_size=(3, 3), activation='relu')(tensor_out)

tensor_out = BatchNormalization(axis=1)(tensor_out)

tensor_out = MaxPooling2D(pool_size=(2, 2))(tensor_out)



tensor_out = Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu')(tensor_out)

tensor_out = Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu')(tensor_out)

tensor_out = MaxPooling2D(pool_size=(2, 2))(tensor_out)



tensor_out = Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu')(tensor_out)

tensor_out = BatchNormalization(axis=1)(tensor_out)

tensor_out = MaxPooling2D(pool_size=(2, 2))(tensor_out)



tensor_out = Flatten()(tensor_out)

#tensor_out = Dropout(0.5)(tensor_out)



tensor_out = [Dense(29, name='digit1', activation='softmax')(tensor_out),\

              Dense(29, name='digit2', activation='softmax')(tensor_out),\

              Dense(29, name='digit3', activation='softmax')(tensor_out),\

              Dense(29, name='digit4', activation='softmax')(tensor_out),\

              Dense(29, name='digit5', activation='softmax')(tensor_out)]



model = Model(inputs=tensor_in, outputs=tensor_out)

model.compile(loss='categorical_crossentropy', optimizer='nadam', metrics=['acc'])

model.summary()
filepath='/kaggle/working/cnn_model.hdf5'

#try:

#    model = load_model(filepath)

#    print('model is loaded...')

#except:

#    model.save(filepath)

#    print('training new model...')

    

#checkpoint = ModelCheckpoint(filepath, monitor='val_digit1_acc', verbose=1, save_best_only=True, mode='max')

earlystop = EarlyStopping(monitor='val_loss', patience=20, verbose=1, mode='auto')

#tensorBoard = TensorBoard(log_dir='../output/logs', histogram_freq = 1, write_graph=True, write_grads=False, write_images=True)

callbacks_list = [earlystop]

history = model.fit(train_data, train_label, batch_size=50, epochs=80, verbose=1, validation_split=0.2, callbacks=callbacks_list, shuffle=True)

model.save(filepath)