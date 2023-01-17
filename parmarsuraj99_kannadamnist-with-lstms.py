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
import matplotlib.pyplot as plt



from sklearn.model_selection import train_test_split



import tensorflow as tf

from tensorflow.keras import Model, Sequential

from tensorflow.keras.layers import Conv2D, Dense, Flatten, LSTM, Dropout

from tensorflow.keras.utils import to_categorical
ROOT_DIR = os.path.relpath('/kaggle/input/Kannada-MNIST/')
train_df = pd.read_csv(os.path.join(ROOT_DIR, 'train.csv'))

test_df = pd.read_csv(os.path.join(ROOT_DIR, 'test.csv'))

dig_mnist = pd.read_csv(os.path.join(ROOT_DIR, 'Dig-MNIST.csv'))

sample_submission = pd.read_csv(os.path.join(ROOT_DIR, 'sample_submission.csv'))
train = train_df.to_numpy()

test = test_df.to_numpy()

dig = dig_mnist.to_numpy()
dig
train.shape
train_labels = train[:, 0]

dig_labels = dig[:, 0]



train_imgs = train[:, 1:]

dig_imgs = dig[:, 1:]



test_id = test[:, 0]

test_imgs = test[:, 1:]
train_labels_oh = to_categorical(train_labels)

valid_labels_dig = to_categorical(dig_labels)
train_imgs = train_imgs.reshape(-1, 28, 28)

test_imgs = test_imgs.reshape(-1, 28, 28)

valid_imgs = dig_imgs.reshape(-1, 28, 28)
train_imgs = train_imgs/255.0

valid_imgs = valid_imgs/255.0

test_imgs = test_imgs/255.0
def _print_with_labels(img_arr, label):

    print(label)

    plt.imshow(img_arr)

    

def show_img(index):

    _print_with_labels(train_imgs[index], train_labels[index])
x_train, x_test, y_train, y_test = train_test_split(train_imgs, train_labels_oh)
y_train.shape
model = Sequential()

model.add(LSTM(128, input_shape=(x_train.shape[1:]), activation='relu', return_sequences=True))

model.add(Dropout(0.2))



model.add(LSTM(128, activation='relu'))

model.add(Dropout(0.2))



model.add(Dense(64, activation='relu'))

model.add(Dropout(0.2))



model.add(Dense(10, activation='softmax'))
model.summary()
opt = tf.keras.optimizers.Adam(lr=1e-3, decay=1e-5)
model.compile(loss='categorical_crossentropy', optimizer=opt,

             metrics=['accuracy'])
hist=model.fit(train_imgs, train_labels_oh, epochs=10, batch_size=64, validation_data=(valid_imgs, valid_labels_dig))
ans = model.predict(test_imgs)
ans_f = np.argmax(ans, 1)
submission = pd.DataFrame.from_dict({'id':test_id, 'label':ans_f})
submission.set_index('id')
submission.to_csv('submission.csv', index=False)
sub = pd.read_csv('submission.csv', index_col=0)
sub