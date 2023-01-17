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
import zipfile



with zipfile.ZipFile('/kaggle/input/dogs-vs-cats-redux-kernels-edition/train.zip') as existing_zip:

    existing_zip.extractall()
print(os.listdir('./train'))
import zipfile



with zipfile.ZipFile('/kaggle/input/dogs-vs-cats-redux-kernels-edition/test.zip') as existing_zip:

    existing_zip.extractall()
print(os.listdir('./test'))
print(os.listdir('./'))
import tensorflow

from tensorflow import keras

%matplotlib inline

import matplotlib.pyplot as plt
%matplotlib inline

import matplotlib.pyplot as plt



import os, cv2, random

import numpy as np

import pandas as pd
from matplotlib import ticker

import seaborn as sns
TRAIN_DIR = "./train/"

TEST_DIR = "./test/"
ROWS = 64

COLS = 64

CHANNELS = 3
train_images = [TRAIN_DIR+i for i in os.listdir(TRAIN_DIR)]
train_images
len(train_images)
train_images[1000]
train_dogs =   [TRAIN_DIR+i for i in os.listdir(TRAIN_DIR) if 'dog' in i]
train_cats =   [TRAIN_DIR+i for i in os.listdir(TRAIN_DIR) if 'cat' in i]
len(train_dogs)
len(train_cats)
test_images =  [TEST_DIR+i for i in os.listdir(TEST_DIR)]
len(test_images)
train_images = train_dogs[:12500] + train_cats[:12500]
len(train_images)
random.shuffle(train_images)
test_images=test_images[:25000]
len(test_images)
def read_image(file_path):

    img = cv2.imread(file_path, cv2.IMREAD_COLOR)#カラー画像読込

    return cv2.resize(img, (ROWS, COLS), interpolation=cv2.INTER_CUBIC)#リサイズ
#画像ファイル群を渡してndArray配列に替える

def prep_data(images):

    count = len(images)

    data = np.ndarray((count, CHANNELS, ROWS, COLS), dtype=np.uint8)#dtype 濃度

    

    for i, image_file in enumerate(images):

        image = read_image(image_file)

        data[i] = image.T #転置行列（縦横逆）

        if i%250 == 0: print('Processed {} of {}'.format(i, count))

            

    return data
train = prep_data(train_images)

test = prep_data(test_images)
train.shape
test.shape
labels = []

for i in train_images:#train_imagesフォルダから　i　を1づつ取り出し、「i」に入れる

    if 'dog.' in i:

        labels.append(1)

    else:

        labels.append(0)
labels[0:25000]
train_images[0:25000]
sns.countplot(labels)

plt.title('Cats and Dogs')
def show_cats_and_dogs(idx):

    cat = read_image(train_cats[idx])

    dog = read_image(train_dogs[idx])

    pair = np.concatenate((cat, dog), axis=1)

    plt.figure(figsize=(10,5))

    plt.imshow(pair)

    plt.show()
for idx in range(100,115):

    show_cats_and_dogs(idx)
from keras.models import Sequential

from keras.layers import Input, Dropout, Flatten, Conv2D, MaxPooling2D, Dense, Activation

from keras.optimizers import RMSprop

from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping

from keras.utils import np_utils
import keras.backend.tensorflow_backend as tfback
import tensorflow as tf
def _get_available_gpus():  



    if tfback._LOCAL_DEVICES is None:  

        devices = tf.config.list_logical_devices()  

        tfback._LOCAL_DEVICES = [x.name for x in devices]  

    return [x for x in tfback._LOCAL_DEVICES if 'device:gpu' in x.lower()]





tfback._get_available_gpus = _get_available_gpus
optimizer = RMSprop(lr=1e-4)

objective = 'binary_crossentropy'



def catdog():

    model = Sequential()

    

    model.add(Conv2D(32, (3, 3), padding='same', input_shape=(3, ROWS, COLS), activation='relu'))

    model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))

    model.add(MaxPooling2D(data_format="channels_first", pool_size=(2, 2)))

    

    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))

    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))

    model.add(MaxPooling2D(data_format="channels_first", pool_size=(2, 2)))

    

    model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))

    model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))

    model.add(MaxPooling2D(data_format="channels_first", pool_size=(2, 2)))

    

    model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))

    model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))

#     model.add(Convolution2D(256, 3, 3, border_mode='same', activation='relu'))

    model.add(MaxPooling2D(data_format="channels_first", pool_size=(2, 2)))

    

    model.add(Flatten())

    model.add(Dense(256, activation='relu'))

    model.add(Dropout(0.5))

    

    model.add(Dense(256, activation='relu'))

    model.add(Dropout(0.5))

    

    model.add(Dense(1))

    model.add(Activation('sigmoid'))



    model.compile(loss=objective, optimizer=optimizer, metrics=['accuracy'])

    return model



model = catdog()
nb_epoch=10

batch_size=250



class LossHistory(Callback):

    def on_train_begin(self, logs={}):

        self.losses = []

        self.val_losses = []

        

    def on_epoch_end(self, batch, logs={}):

        self.losses.append(logs.get('loss'))

        self.val_losses.append(logs.get('val_loss'))



# monitor: 監視対象

# patience: 訓練が停止し，値が改善しなくなってからのエポック数．

early_stopping = EarlyStopping(monitor='val_loss', patience=3, verbose=1, mode='auto')



def run_catdog():

    

    history = LossHistory()

    model.fit(train, labels, batch_size=batch_size, epochs=nb_epoch,

              validation_split=0.25, verbose=0, shuffle=True, callbacks=[history, early_stopping])

    



    predictions = model.predict(test, verbose=0)

    return predictions, history



predictions, history = run_catdog()

# タプルで受け取る

loss = history.losses

val_loss = history.val_losses



plt.xlabel('Epochs')

plt.ylabel('Loss')



plt.title('CatdogNet Loss Trend')

plt.plot(loss, 'blue', label='Training Loss')

plt.plot(val_loss, 'green', label='Validation Loss')

plt.xticks(range(0,nb_epoch)[0::2])

plt.legend()

plt.show()
print(len(predictions))
for i in range(0,10):

    if predictions[i, 0] >= 0.5:

        print('I am {:.2%} sure this is a Dog'.format(predictions[i][0]))

    else:

        print('I am {:.2%} sure this is a Cat'.format(1-predictions[i][0]))

        

    plt.imshow(test[i].T)

    plt.show()
test_pred=model.predict(test, verbose=0)

print(test_pred)
test_pred = (test_pred > 0.5).astype(int)

test_pred[0:]
len(test_pred)
print(os.listdir('/kaggle/input/dogs-vs-cats-redux-kernels-edition/'))
sample=pd.read_csv('/kaggle/input/dogs-vs-cats-redux-kernels-edition/sample_submission.csv')
submit11=sample[["id"]].copy()

submit11['label']=test_pred
submit11.to_csv(path_or_buf="submit11.csv", sep=",", index=False,header=False)
submit11