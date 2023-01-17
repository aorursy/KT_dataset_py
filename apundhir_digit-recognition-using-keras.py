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
train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")
train.info()
train.head()
Y_Train = train["label"]

X_Train = train.drop(labels = ["label"], axis = 1)



X_Train = X_Train.values.reshape(-1,28,28,1)

X_Test = test.values.reshape(-1,28,28,1)
print("Number of records in training data "+str(X_Train.shape[0]))

print("Number of records in test data "+str(test.shape[0]))
import matplotlib.pyplot as plt

import random



plt.figure(figsize=(10,5))

for i in range(10):  

    plt.subplot(1, 10, i+1)

    r = random.randint(0, X_Train.shape[0])

    plt.imshow(X_Train[r].reshape((28,28)),cmap=plt.cm.binary)

    plt.axis('off')
## Normalization

X_Train = X_Train / 255.0

X_Test = X_Test / 255.0
# Augmentation

image = X_Train[r].reshape((28,28))

plt.imshow(image)
from skimage import transform as tf



# specify x and y coordinates to be used for shifting (mid points)

shift_x, shift_y = image.shape[0]/2, image.shape[1]/2



# translation by certain units

matrix_to_topleft = tf.SimilarityTransform(translation=[-shift_x, -shift_y])

matrix_to_center = tf.SimilarityTransform(translation=[shift_x, shift_y])





# rotation

rot_transforms =  tf.AffineTransform(rotation=np.deg2rad(15))

rot_matrix = matrix_to_topleft + rot_transforms + matrix_to_center

rot_image = tf.warp(image, rot_matrix)



# scaling 

scale_transforms = tf.AffineTransform(scale=(1.5, 1.5))

scale_matrix = matrix_to_topleft + scale_transforms + matrix_to_center

scale_image_zoom_out = tf.warp(image, scale_matrix)



scale_transforms = tf.AffineTransform(scale=(0.8, 0.8))

scale_matrix = matrix_to_topleft + scale_transforms + matrix_to_center

scale_image_zoom_in = tf.warp(image, scale_matrix)



# shear transforms

shear_transforms = tf.AffineTransform(shear=np.deg2rad(20))

shear_matrix = matrix_to_topleft + shear_transforms + matrix_to_center

shear_image = tf.warp(image, shear_matrix)
items = [image, rot_image, scale_image_zoom_out, scale_image_zoom_in, shear_image]



f, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1, 5, sharey=True)

f.set_figwidth(15)

ax1.imshow(image)

ax2.imshow(rot_image)

ax3.imshow(scale_image_zoom_out)

ax4.imshow(scale_image_zoom_in)

ax5.imshow(shear_image)
from keras.preprocessing.image import ImageDataGenerator



datagen = ImageDataGenerator(

        rotation_range=15,  ## Degree range for random rotations.

        zoom_range = 0.10,  ## Range for random zoom

        width_shift_range=0.1, ## fraction of total width

        height_shift_range=0.1, ## fraction of total height

        shear_range=0.1,

)



val_datagen = ImageDataGenerator()
## Bring target to categorical

from keras.utils.np_utils import to_categorical



labels = to_categorical(Y_Train, num_classes = 10)
from keras.models import Sequential

from keras.layers import Conv2D, Flatten, MaxPool2D, Dropout, BatchNormalization, Dense



model = Sequential()



model.add(Conv2D(32, kernel_size = 3, activation='relu', input_shape = (28, 28, 1)))

model.add(BatchNormalization())

model.add(Conv2D(32, kernel_size = 3, activation='relu'))

model.add(BatchNormalization())

model.add(Dropout(0.2))



model.add(Conv2D(64, kernel_size = 3, activation='relu'))

model.add(BatchNormalization())

model.add(Conv2D(64, kernel_size = 3, activation='relu'))

model.add(BatchNormalization())





model.add(Conv2D(128, kernel_size = 3, activation='relu'))

model.add(BatchNormalization())

model.add(Conv2D(128, kernel_size = 3, activation='relu'))

model.add(BatchNormalization())

model.add(Dropout(0.4))



model.add(Conv2D(128, kernel_size = 3, activation='relu'))

model.add(BatchNormalization())

model.add(Flatten())

model.add(Dropout(0.2))

model.add(Dense(10, activation='softmax'))



model.summary()
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
batch_size = 50

epochs = 50



train_generator = datagen.flow(

    X_Train,

    labels,

    batch_size=batch_size

)
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau



model_name = 'model' + '/'

    

if not os.path.exists(model_name):

    os.mkdir(model_name)

        

filepath = model_name + 'model.h5'



checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=False, save_weights_only=False, mode='auto', period=1)



LR = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, cooldown=1, verbose=1)

callbacks_list = [checkpoint, LR]
## Lets keep 10% of the data for validation

from sklearn.model_selection import train_test_split



X_train, X_val, Y_train, Y_val = train_test_split(X_Train, labels, test_size = 0.1)



model_hist = model.fit_generator(train_generator, steps_per_epoch=len(X_train)/batch_size, epochs=epochs, verbose=1, 

                    callbacks=callbacks_list,

                    validation_data = (X_val,Y_val), class_weight=None, workers=1, initial_epoch=0)
plt.plot(model_hist.history['loss'])

plt.plot(model_hist.history['val_loss'])

plt.title('Model Loss')

plt.ylabel('Loss')

plt.xlabel('Epochs')

plt.legend(['train', 'test'])

plt.show()
X_Test.shape
predicted_digits = model.predict(X_Test).argmax(axis=-1)

result_df = pd.DataFrame()

result_df['ImageId'] = list(range(1,X_Test.shape[0] + 1))

result_df['Label'] = predicted_digits

result_df.to_csv("submission.csv", index = False)