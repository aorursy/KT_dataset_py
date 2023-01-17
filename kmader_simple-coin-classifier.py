from glob import glob

import os

import pandas as pd

from skimage.io import imread

from scipy.ndimage import zoom

import matplotlib.pyplot as plt

    

from sklearn.preprocessing import LabelEncoder

from keras.utils.np_utils import to_categorical

import numpy as np

def imread_size(in_path):

    t_img = imread(in_path)

    return zoom(t_img, [96/t_img.shape[0], 96/t_img.shape[1]]+([1] if len(t_img.shape)==3 else []),

               order = 2)
base_img_dir = os.path.join('..', 'input')

all_training_images = glob(os.path.join(base_img_dir, '*', '*.png'))

full_df = pd.DataFrame(dict(path = all_training_images))

full_df['category'] = full_df['path'].map(lambda x: os.path.basename(os.path.dirname(x)))

full_df = full_df.query('category != "valid"')

cat_enc = LabelEncoder()

full_df['category_id'] = cat_enc.fit_transform(full_df['category'])

y_labels = to_categorical(np.stack(full_df['category_id'].values,0))

print(y_labels.shape)

full_df['image'] = full_df['path'].map(imread_size)

full_df.sample(3)

full_df['category'].value_counts()
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(np.expand_dims(np.stack(full_df['image'].values,0),-1), 

                                                    y_labels,

                                   random_state = 12345,

                                   train_size = 0.75,

                                   stratify = full_df['category'])

print('Training Size', X_train.shape)
from keras.preprocessing.image import ImageDataGenerator # (docu: https://keras.io/preprocessing/image/)



train_datagen = ImageDataGenerator(

        samplewise_std_normalization = True,

        shear_range=0.2,

        zoom_range=0.2,

        rotation_range = 360,

        )



test_datagen = ImageDataGenerator(

        samplewise_std_normalization = True)



train_gen = train_datagen.flow(X_train, y_train, batch_size=32)



test_gen = train_datagen.flow(X_test, y_test, batch_size=200)

fig, (ax1, ax2) = plt.subplots(2, 4, figsize = (12, 6))

for c_ax1, c_ax2, (train_img, _), (test_img, _) in zip(ax1, ax2, train_gen, test_gen):

    c_ax1.imshow(train_img[0,:,:,0])

    c_ax1.set_title('Train Image')

    

    c_ax2.imshow(test_img[0,:,:,0])

    c_ax2.set_title('Test Image')
from keras.models import Sequential

from keras.layers import Conv2D, Dense, BatchNormalization, Flatten

simple_cnn = Sequential()

simple_cnn.add(Conv2D(filters = 8, kernel_size = (3,3), strides = (2,2), input_shape = (96, 96, 1)))

simple_cnn.add(Conv2D(filters = 8, kernel_size = (3,3), strides = (2,2)))

simple_cnn.add(Conv2D(filters = 8, kernel_size = (3,3), strides = (2,2)))

simple_cnn.add(Flatten())

simple_cnn.add(BatchNormalization())

simple_cnn.add(Dense(16, activation = 'relu'))

simple_cnn.add(Dense(y_labels.shape[1], activation = 'softmax'))

simple_cnn.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['acc'])

loss_history = []

simple_cnn.summary()


for i in range(10):

    loss_history += [simple_cnn.fit_generator(train_gen, steps_per_epoch=10,

                         validation_data=test_gen, validation_steps=1)]
epich = np.cumsum(np.concatenate(

    [np.linspace(0.5, 1, len(mh.epoch)) for mh in loss_history]))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

_ = ax1.plot(epich,

             np.concatenate([mh.history['loss'] for mh in loss_history]),

             'b-',

             epich, np.concatenate(

        [mh.history['val_loss'] for mh in loss_history]), 'r-')

ax1.legend(['Training', 'Validation'])

ax1.set_title('Loss')



_ = ax2.plot(epich, np.concatenate(

    [mh.history['acc'] for mh in loss_history]), 'b-',

                 epich, np.concatenate(

        [mh.history['val_acc'] for mh in loss_history]),

                 'r-')

ax2.legend(['Training', 'Validation'])

ax2.set_title('Accuracy')
from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix

for _, (c_x, c_y) in zip(range(1), test_gen):

    y_pred  = simple_cnn.predict(c_x)

    print(classification_report(np.argmax(c_y,1), 

                                np.argmax(y_pred, 1), target_names = cat_enc.classes_))
c_x.shape