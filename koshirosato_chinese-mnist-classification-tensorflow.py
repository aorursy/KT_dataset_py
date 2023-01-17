import os

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from skimage import io, transform

from sklearn.model_selection import train_test_split

import tensorflow as tf

import tensorflow_addons as tfa

from tensorflow.keras.layers import *

from tensorflow.keras.models import Sequential

from tqdm import tqdm
SEED = 42

EPOCHS = 50

BATCH_SIZE = 32

IMG_SIZE = 64

IMG_ROOT = '../input/chinese-mnist/data/data/'



train_df = pd.read_csv('../input/chinese-mnist/chinese_mnist.csv')
def seed_everything(seed):

    np.random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    tf.random.set_seed(seed)



seed_everything(SEED)
train_df
train_df.isnull().sum()
train_df['character'].value_counts()
def create_file_name(x):

    file_name = f'input_{x[0]}_{x[1]}_{x[2]}.jpg'

    return file_name





def add_filenames(df, img_root):

    filenames = list(os.listdir(img_root))

    df['filenames'] = df.apply(create_file_name, axis=1)

    return df 

    

train_df = add_filenames(train_df, IMG_ROOT)

train_df
train_df, test_df = train_test_split(train_df, 

                                     test_size=0.2,

                                     random_state=SEED,

                                     stratify=train_df['character'].values) 

train_df, val_df = train_test_split(train_df,

                                    test_size=0.1,

                                    random_state=SEED,

                                    stratify=train_df['character'].values)
def create_datasets(df, img_root, img_size, n):

    imgs = []

    for filename in tqdm(df['filenames']):

        img = io.imread(img_root+filename)

        img = transform.resize(img, (img_size,img_size,n))

        imgs.append(img)

        

    imgs = np.array(imgs)

    df = pd.get_dummies(df['character'])

    return imgs, df





train_imgs, train_df = create_datasets(train_df, IMG_ROOT, IMG_SIZE, 1)

val_imgs, val_df = create_datasets(val_df, IMG_ROOT, IMG_SIZE, 1)

test_imgs, test_df = create_datasets(test_df, IMG_ROOT, IMG_SIZE, 1)
input_shape = (IMG_SIZE, IMG_SIZE, 1)



model = Sequential()

model.add(Conv2D(16, kernel_size=3, padding='same', input_shape=input_shape, activation='relu'))

model.add(Conv2D(16, kernel_size=3, padding='same', activation='relu'))

model.add(MaxPool2D(3))

model.add(Dropout(0.2))

model.add(Flatten())

model.add(Dense(15, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])





model.summary()
es_callback = tf.keras.callbacks.EarlyStopping(patience=20, 

                                               verbose=1, 

                                               restore_best_weights=True)





history = model.fit(train_imgs, 

                    train_df, 

                    batch_size=BATCH_SIZE, 

                    epochs=EPOCHS, 

                    callbacks=[es_callback],

                    validation_data=(val_imgs, val_df))



pd.DataFrame(history.history)[['accuracy', 'val_accuracy']].plot()

pd.DataFrame(history.history)[['loss', 'val_loss']].plot()

plt.show()
model.evaluate(test_imgs, test_df) 
model = Sequential()

model.add(Conv2D(16, kernel_size=3, padding='same', input_shape=input_shape, activation='relu'))

model.add(Conv2D(16, kernel_size=3, padding='same', activation='relu'))

model.add(BatchNormalization())

model.add(MaxPool2D(3))

model.add(Dropout(0.2))

model.add(Conv2D(16, kernel_size=3, padding='same', activation='relu'))

model.add(Conv2D(16, kernel_size=3, padding='same', activation='relu'))

model.add(BatchNormalization())

model.add(MaxPool2D(3))

model.add(Dropout(0.2))

model.add(Flatten())

model.add(Dense(15, activation='softmax'))

opt = tfa.optimizers.LazyAdam()

loss = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.025)

model.compile(optimizer=opt, loss=loss, metrics=['accuracy'])





model.summary()
def get_lr_callback(batch_size=32, plot=False):

    lr_start   = 0.003

    lr_max     = 0.00125 * batch_size

    lr_min     = 0.001

    lr_ramp_ep = 20

    lr_sus_ep  = 0

    lr_decay   = 0.8

   

    def lrfn(epoch):

        if epoch < lr_ramp_ep:

            lr = (lr_max - lr_start) / lr_ramp_ep * epoch + lr_start

            

        elif epoch < lr_ramp_ep + lr_sus_ep:

            lr = lr_max

            

        else:

            lr = (lr_max - lr_min) * lr_decay**(epoch - lr_ramp_ep - lr_sus_ep) + lr_min

            

        return lr



    lr_callback = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=False)

    if plot == True:

        rng = [i for i in range(EPOCHS)]

        y = [lrfn(x) for x in rng]

        plt.plot(rng, y)

        plt.xlabel('epoch', size=14); plt.ylabel('learning_rate', size=14)

        plt.title('Training Schedule', size=16)

        plt.show()

    return lr_callback



get_lr_callback(plot=True)
es_callback = tf.keras.callbacks.EarlyStopping(patience=20, 

                                               verbose=1, 

                                               restore_best_weights=True)





history = model.fit(train_imgs, 

                    train_df, 

                    batch_size=BATCH_SIZE, 

                    epochs=EPOCHS, 

                    callbacks=[es_callback, get_lr_callback(BATCH_SIZE)],

                    validation_data=(val_imgs, val_df))



pd.DataFrame(history.history)[['accuracy', 'val_accuracy']].plot()

pd.DataFrame(history.history)[['loss', 'val_loss']].plot()

plt.show()
model.evaluate(test_imgs, test_df) 