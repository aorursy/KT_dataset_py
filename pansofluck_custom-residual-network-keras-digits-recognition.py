# first load libraries

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import random

import tensorflow as tf

import os

import math

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

from tensorflow.keras.utils import to_categorical

from tensorflow.keras.optimizers import RMSprop

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.model_selection import train_test_split





import albumentations


def datagen(features, labels, batch_size, p=1):

    aug = albumentations.Compose([

          albumentations.GaussianBlur(p=0.01),

          albumentations.ShiftScaleRotate(p=0.5),

          albumentations.Rotate(limit=20, p=0.5),

          albumentations.OpticalDistortion(p=0.1),

          albumentations.ImageCompression(p=0.05)], p=p)

    

    batch_features = np.zeros((batch_size, features.shape[1], features.shape[2], features.shape[3]))

    batch_labels = np.zeros((batch_size, labels.shape[1]))

    batches = 0

    finish = len(features) / batch_size

    while True:

        # Fill arrays of batch size with augmented data taken randomly from full passed arrays

        indexes = random.sample(range(len(features)), batch_size)

        # Perform the exactly the same augmentation for X and y

        random_augmented_images = [aug(image=x.reshape(28,28,1))['image'] for x in features[indexes]]

        random_augmented_labels = [x for x in labels[indexes]]

        batch_features[:,:,:,:] = random_augmented_images

        batch_labels[:,:] = random_augmented_labels



#         if batches >= finish:

#             # we need to break the loop by hand because

#             # the generator loops indefinitely

#             break

#         batches += 1

        

        yield batch_features, batch_labels
# for reproducible results :

def seed_everything(seed=13):

    np.random.seed(seed)

    tf.random.set_seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    os.environ['TF_DETERMINISTIC_OPS'] = '1'

    os.environ['TF_KERAS'] = '1'

    random.seed(seed)

    

seed_everything(42)
try :

    tpu=tf.distribute.cluster_resolver.TPUClusterResolver()

    print('Running on :',tpu.master())

except ValueError :

    tpu = None



if tpu :    

    tf.config.experimental_connect_to_cluster(tpu)

    tf.tpu.experimental.initialize_tpu_system(tpu)

    strategy = tf.distribute.experimental.TPUStrategy(tpu)

else :

    strategy = tf.distribute.get_strategy()

    

print('Replicas :',strategy.num_replicas_in_sync)  
# load data

df_train = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')

df_test = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')
# split our data into features & target

X_train = df_train.drop('label', axis=1).values

y_train = df_train['label'].values.reshape(-1,1)



X_test = df_test.values
# rescale variables

X_train = X_train.astype('float32') / 255

X_test = X_test.astype('float32') / 255
# check first few images

AUGMENTATIONS = albumentations.Compose([

                albumentations.GaussianBlur(p=0.01),

                albumentations.ShiftScaleRotate(p=0.5),

                albumentations.Rotate(limit=20, p=0.5),

                albumentations.OpticalDistortion(p=0.1),

                albumentations.ImageCompression(p=0.05)], p=1)



plt.figure(figsize=(15,15))

for i in range(25):

    plt.subplot(5,5,i+1)

    im = AUGMENTATIONS(image=X_train[i].reshape(28,28,1))['image']

#     im = X_train[i]

#     print(im)

    plt.imshow(im.reshape(28,28), cmap='gray')

    plt.title('Number:' + str(y_train[i][0]))

    plt.axis('off')
# reshape features for tensorflow

X_train = X_train.reshape(-1,28,28,1)

X_test = X_test.reshape(-1,28,28,1)



# one hot encode for target variable

y_train = to_categorical(y_train)

target_count = y_train.shape[1]



X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)
print(X_train.shape)

print(y_train.shape)
train_gen = datagen(X_train, y_train, 128, p=1)

valid_gen = datagen(X_val, y_val, 128, p=0)



initializer = tf.keras.initializers.GlorotUniform()



inputs = tf.keras.Input(shape=(28, 28, 1))

x = tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), padding='same', activation='relu', kernel_initializer=initializer)(inputs)

x = tf.keras.layers.BatchNormalization()(x)

x = tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), padding='same', activation='relu', kernel_initializer=initializer)(x)

x = tf.keras.layers.BatchNormalization()(x)

x = tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), padding='same', activation='relu', kernel_initializer=initializer)(x)

x = tf.keras.layers.BatchNormalization()(x)



y1 = tf.keras.layers.Conv2D(filters=64, kernel_size=(1,1), padding='same', activation='relu', kernel_initializer=initializer)(inputs)

y1 = tf.keras.layers.BatchNormalization()(y1)

x = tf.keras.layers.Add()([x, y1])



x = tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), padding='valid', strides=(2,2), activation='relu', kernel_initializer=initializer)(x)

x = tf.keras.layers.Dropout(0.25)(x)

second_block = tf.keras.layers.BatchNormalization()(x)



x = tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), padding='same', activation='relu', kernel_initializer=initializer)(x)

x = tf.keras.layers.BatchNormalization()(x)

x = tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), padding='same', activation='relu', kernel_initializer=initializer)(x)

x = tf.keras.layers.BatchNormalization()(x)

x = tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), padding='same', activation='relu', kernel_initializer=initializer)(x)

x = tf.keras.layers.BatchNormalization()(x)



y1 = tf.keras.layers.Conv2D(filters=128, kernel_size=(1,1), padding='same', activation='relu', kernel_initializer=initializer)(second_block)

y1 = tf.keras.layers.BatchNormalization()(y1)

x = tf.keras.layers.Add()([x, y1])



x = tf.keras.layers.Flatten()(x)

x = tf.keras.layers.Dense(256, activation='relu', kernel_initializer=initializer)(x)

x = tf.keras.layers.Dropout(0.5)(x)

x = tf.keras.layers.BatchNormalization()(x)

outputs = tf.keras.layers.Dense(10, activation='softmax', kernel_initializer=initializer)(x)



model = tf.keras.Model(inputs=inputs, outputs=outputs)



optimizer = RMSprop(learning_rate=0.01,rho=0.99)

model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])



reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.3, verbose=1,patience=2, min_lr=0.0000001)



mc = ModelCheckpoint('best_model.h5', monitor = 'val_loss' , mode = 'min', verbose = 1 , save_best_only = True)



callback = EarlyStopping(monitor='loss', patience=5)
from tensorflow.keras.utils import plot_model

# model.summary()

plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
model.fit_generator(

    train_gen,

    steps_per_epoch=int(len(X_train)/128),

    epochs=50,

    validation_data=valid_gen,

    validation_steps=int(len(X_val)/128),

    callbacks=[callback, reduce_lr, mc])
model.load_weights('best_model.h5')

y_test_hat = model.predict(X_test).argmax(axis=1)



df_submission = pd.read_csv('/kaggle/input/digit-recognizer/sample_submission.csv')

df_submission['Label'] = y_test_hat.astype('int32')

df_submission.to_csv('submission.csv', index=False)

print('Submission saved!')
import gc

del model

gc.collect()