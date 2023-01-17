import pandas as pd

import os,cv2

from tqdm import tqdm_notebook

from IPython.display import Image

from keras.preprocessing import image

from keras import optimizers

from keras.layers import Conv2D, BatchNormalization, Dense, MaxPooling2D, Dropout, Flatten, GlobalAveragePooling2D

from keras.models import Sequential

from keras.applications.imagenet_utils import preprocess_input

import matplotlib.pyplot as plt

import seaborn as sns

from keras import regularizers

from keras.preprocessing.image import ImageDataGenerator

from keras.applications.vgg16 import VGG16

print(os.listdir("../input"))



import numpy as np
# importing train file with Img_ID & has_cactus column

train = pd.read_csv('../input/train.csv')

train.shape
df_test=pd.read_csv('../input/sample_submission.csv')
# printing some of the columns 

train.head()
# printing some info about the dataset

train.info()
train.has_cactus.value_counts().plot.bar()

print('{:.2f} of images have has_cactus label=0'.format((train.has_cactus.value_counts()[0]/train.shape[0])*100))

print('{:.2f} of images have has_cactus label=1'.format((train.has_cactus.value_counts()[1]/train.shape[0])*100))
# setting train & test directory paths for importing images

train_dir="../input/train/train"

test_dir="../input/test/test"
# using ImageDataGenerator, available in keras for preprocessing

datagen=ImageDataGenerator(rescale=1./255)

batch_size=150
# converting the datatype has_cactus column to str 

train.has_cactus=train.has_cactus.astype(str)
# splitting the train dataset into train(15000) and validation(2500)

train_generator=datagen.flow_from_dataframe(dataframe=train[:15001],directory=train_dir,x_col='id',

                                            y_col='has_cactus',class_mode='binary',batch_size=batch_size,

                                            target_size=(32, 32))





validation_generator=datagen.flow_from_dataframe(dataframe=train[15000:],directory=train_dir,x_col='id',

                                                y_col='has_cactus',class_mode='binary',batch_size=50,

                                                target_size=(32, 32))
model = Sequential()

# layer 1

model.add(Conv2D(64, (3,3), padding='same', activation="relu", input_shape=(32, 32, 3)))

model.add(BatchNormalization())

# layer 2

model.add(Conv2D(64, (3,3), padding='same', activation="relu"))

model.add(BatchNormalization())

# layer 3

model.add(Conv2D(64, (3,3), padding='same', activation="relu"))

model.add(BatchNormalization())

model.add(MaxPooling2D(2,2))

model.add(Dropout(0.4))

# layer 4

model.add(Conv2D(128, (3,3), padding='same', activation="relu"))

model.add(BatchNormalization())

# layer 5

model.add(Conv2D(128, (3,3), padding='same', activation="relu"))

model.add(BatchNormalization())

# layer 5

model.add(Conv2D(128, (3,3), padding='same', activation="relu"))

model.add(BatchNormalization())

model.add(MaxPooling2D(2,2))

model.add(Dropout(0.4))

# layer 6

model.add(Conv2D(256, (3,3), padding='same', activation="relu"))

model.add(BatchNormalization())

# layer 7

model.add(Conv2D(256, (3,3), padding='same', activation="relu"))

model.add(BatchNormalization())

model.add(MaxPooling2D(2,2))

model.add(Dropout(0.4))

# layer 8

model.add(Conv2D(256, (3,3), padding='same', activation="relu"))

model.add(BatchNormalization())

# layer 9

model.add(Conv2D(256, (3,3), padding='same', activation="relu"))

model.add(BatchNormalization())

model.add(MaxPooling2D(2,2))

model.add(Dropout(0.4))



model.add(GlobalAveragePooling2D())

model.add(Dense(units=256, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(units=256, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(units=1, activation='sigmoid'))
model.summary()
model.compile(loss='binary_crossentropy',optimizer=optimizers.Adam(lr=0.001),metrics=['acc'])

epochs=10

history=model.fit_generator(train_generator,steps_per_epoch=100,epochs=10,

                            validation_data=validation_generator,validation_steps=50)
acc=history.history['acc']  ##getting  accuracy of each epochs

epochs_=range(0,epochs)    

plt.plot(epochs_,acc,label='training accuracy')

plt.xlabel('no of epochs')

plt.ylabel('accuracy')



acc_val=history.history['val_acc']  ##getting validation accuracy of each epochs

plt.plot(epochs_,acc_val,label='validation accuracy')

plt.title("no of epochs vs accuracy")

plt.legend()
test_df = pd.read_csv(os.path.join('../input', "sample_submission.csv"))

print(test_df.head())

test_images = []

images = test_df['id'].values



for image_id in images:

    test_images.append(cv2.imread(os.path.join(test_dir, image_id)))

    

test_images = np.asarray(test_images)

test_images = test_images / 255.0

print("Number of Test set images: " + str(len(test_images)))
y_pred = model.predict(test_images)
test_df['has_cactus'] = y_pred

test_df.to_csv('submission.csv', index = False)