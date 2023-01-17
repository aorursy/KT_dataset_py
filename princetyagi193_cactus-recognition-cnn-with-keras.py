import numpy as np 

import pandas as pd

import tensorflow as tf

import keras

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

from keras.models import Sequential

from keras.layers import Conv2D, MaxPooling2D,DepthwiseConv2D, Dense, Flatten, Dropout, BatchNormalization, LeakyReLU,GlobalAveragePooling2D, Activation, Average,AveragePooling2D

import matplotlib.pyplot as plt

from keras.optimizers import Adam,Adamax

from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint, TensorBoard

import seaborn as sns

import os

print(os.listdir("../input"))
bas_dir=os.path.join('..','input')

csv_train=pd.read_csv(os.path.join(bas_dir,'train.csv'))

train_dir=os.path.join(bas_dir,'train/train')

test_dir=os.path.join(bas_dir,'test/test')
csv_train.head()
'''count=csv_train['has_cactus'].value_counts()

print(count)

count.plot(kind="bar")

plt.show()'''

sns.countplot(x = 'has_cactus',

              data = csv_train,

              order = csv_train['has_cactus'].value_counts().index)

plt.show()
# Using Image Generator to preprocess the data

csv_train['has_cactus']=csv_train['has_cactus'].astype('str')

batch_size = 100

train_size = 15750

validation_size = 1750



datagen = ImageDataGenerator(

    rescale=1./255,

    horizontal_flip=True,

    vertical_flip=True,

    validation_split=0.1,

    zoom_range =0.3,

    zca_whitening = False

    )



data_args = {

    "dataframe": csv_train,

    "directory": train_dir,

    "x_col": 'id',

    "y_col": 'has_cactus',

    "featurewise_center" : True,

    "featurewise_std_normalization" : True,

    "samplewise_std_normalization" : False,

    "samplewise_center" : False,

    "shuffle": True,

    "target_size": (32, 32),

    "batch_size": batch_size,

    "class_mode": 'binary'

}



train_generator = datagen.flow_from_dataframe(**data_args, subset='training')

validation_generator = datagen.flow_from_dataframe(**data_args, subset='validation')


# Plotting Images

def plot_images():

    f,ax=plt.subplots(3,20,figsize=(10,10))

    print(ax.shape)

    for j in range(ax.shape[0]):

        for i  in range(ax.shape[1]):

            image = next(train_generator)

            img = array_to_img(image[0][0])

            ax[j][i].imshow(img)

            ax[j][i].axis("off")

    plt.subplots_adjust(wspace=0, hspace=1)

    plt.show()

    plt.axis("off")

    plt.title("Images", fontsize=18)

    

plot_images()
def Le_Conv():

    model = Sequential()

    model.add(Conv2D(6, kernel_size=(3, 3), strides=(1, 1), activation='relu', input_shape=(32, 32, 3), padding="same"))

    model.add(AveragePooling2D(pool_size=(2, 2), strides=(1, 1), padding='valid'))

    model.add(Conv2D(16, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='valid'))

    model.add(AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))

    model.add(Conv2D(120, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='valid'))

    model.add(Flatten())

    model.add(Dense(84, activation='relu'))

    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer =  Adamax(lr =0.001) , loss = "binary_crossentropy", metrics=["acc"])

    return model
model_history=Le_Conv().fit_generator(train_generator,

              validation_data=validation_generator,

              steps_per_epoch=train_size//batch_size,

              validation_steps=validation_size//batch_size,

              epochs=100, verbose=1, 

              shuffle=True)
test_df = pd.read_csv(os.path.join(bas_dir, "sample_submission.csv"))

test_df.head()
test_df = pd.read_csv(os.path.join(bas_dir, "sample_submission.csv"))

import cv2

test_images = []

images = test_df['id'].values

for image_id in images:

    test_images.append(cv2.imread(os.path.join(test_dir, image_id)))

    

test_images = np.asarray(test_images)

test_images = test_images / 255.0

print("Number of Test set images: " + str(len(test_images)))
predict = Le_Conv().predict(test_images)
test_df['has_cactus'] = predict

test_df.round(2)

test_df.to_csv('aerial-cactus-submission.csv', index = False)