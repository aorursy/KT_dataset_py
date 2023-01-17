import cv2

import os

import pandas as pd 

import matplotlib.pyplot as plt

import matplotlib.image as mpimg

import numpy as np

from skimage import io, transform

from PIL import Image



from sklearn.model_selection import train_test_split



import tensorflow as tf

from tensorflow.keras.callbacks import ReduceLROnPlateau

from tensorflow.keras.applications import ResNet152V2

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Dropout

from tensorflow.keras.callbacks import ModelCheckpoint

from tensorflow.keras.optimizers import Adam, RMSprop

from tensorflow.keras.preprocessing.image import ImageDataGenerator



# create model

from tensorflow.keras.layers import Conv2D, Flatten, Dense, Activation, BatchNormalization, MaxPooling2D, Dropout
IMAGE_WIDTH=100

IMAGE_HEIGHT=100

IMAGE_SIZE=(IMAGE_WIDTH, IMAGE_HEIGHT)

IMAGE_CHANNELS=3

BATCH_SIZE=32

EPOCHS=3

PATH1='../input/helmet/Train_positive/Train_positive/'

PATH2='../input/helmet/Train_negative/Train_negative/'
# train dataset

filenames_helmet_yes = os.listdir(PATH1)

filenames_helmet_no = os.listdir(PATH2)



filenames_list_yes = []

filenames_list_no = []

categories_yes = []

categories_no = []



for filename in filenames_helmet_yes:

    filenames_list_yes.append(PATH1 + filename)

    categories_yes.append(str(1))

for filename in filenames_helmet_no:

    filenames_list_no.append(PATH2 + filename)

    categories_no.append(str(0))

    



df_yes = pd.DataFrame({

    'image': filenames_list_yes,

    'category': categories_yes

})

df_no = pd.DataFrame({

    'image': filenames_list_no,

    'category': categories_no

})

print(df_yes.shape, df_no.shape)

df = df_yes.append(df_no, ignore_index=True)

print(df['image'][0])
#split data into train and valid set

train_df, valid_df = train_test_split(df, test_size = 0.15, stratify = df['category'], random_state = 3)

train_df = train_df.reset_index(drop=True)

valid_df = valid_df.reset_index(drop=True)

total_train = train_df.shape[0]

total_valid = valid_df.shape[0]

print(train_df.shape)

print(valid_df.shape)

#We'll perform individually on train and validation set.

train_datagen = ImageDataGenerator(

                                   rescale=1./255,

                                   )



train_gen = train_datagen.flow_from_dataframe(train_df,

                                              x_col = 'image',

                                              y_col = 'category',

                                              target_size = IMAGE_SIZE,

                                              batch_size = BATCH_SIZE,

                                              class_mode='binary',

                                              validate_filenames=False

                                             )



#we do not augment validation data.

validation_datagen = ImageDataGenerator(rescale=1./255)

valid_gen = validation_datagen.flow_from_dataframe(

    valid_df, 

    x_col="image",

    y_col="category",

    target_size=IMAGE_SIZE,

    class_mode='binary',

    batch_size=BATCH_SIZE,

    validate_filenames=False

)
def create_model():

    model = Sequential()

    model.add(Conv2D(64, (3,3), activation='relu', strides=(2,2), input_shape=(IMAGE_WIDTH,IMAGE_HEIGHT,IMAGE_CHANNELS)))

    model.add(Conv2D(64, (3,3), activation='relu'))

    model.add(Dropout(0.2))

    model.add(Conv2D(64, (3,3), activation='relu'))

    model.add(MaxPooling2D((2, 2)))

    model.add(Dropout(0.2))

    model.add(Conv2D(64, (3,3), activation='relu'))

    model.add(MaxPooling2D((2, 2)))

    model.add(Dropout(0.2))

    model.add(Conv2D(126, (3,3), activation='relu'))

    model.add(MaxPooling2D((2, 2)))

    model.add(Dropout(0.2))

    model.add(Conv2D(126, (3,3), activation='relu'))

    model.add(Dropout(0.3))

    model.add(Flatten())

    model.add(Dense(256, activation='relu'))

    model.add(Dropout(0.3))

    model.add(Dense(1, activation = 'sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

    return model
model = create_model()
model.summary()
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 

                                            patience=2, 

                                            verbose=1, 

                                            factor=0.5, 

                                            min_lr=0.00001)

checkpointer = ModelCheckpoint(filepath = 'mask.weights.best.hdf5', save_best_only = True, save_weights_only = True)

callbacks = [learning_rate_reduction, checkpointer]
model.fit_generator(train_gen,

                    epochs = EPOCHS,

                    validation_data = valid_gen,

                    validation_steps=total_valid//BATCH_SIZE,

                    steps_per_epoch=total_train//BATCH_SIZE,

                    callbacks = callbacks)
loss = pd.DataFrame(model.history.history)

loss[['loss', 'val_loss']].plot()

loss[['acc', 'val_acc']].plot()
PATH3='../input/construction-worker/construction worker/construction workerx8.png'
# face detection with opencv

def face_detection(img):

    

    face_cascade = cv2.CascadeClassifier('../input/haar-cascades-for-face-detection/haarcascade_frontalface_default.xml')

    

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray)

        

    for (x,y,w,h) in faces:

#         cv2.rectangle(img,(x-20,y-20),((x+w)+20,(y+h)+20),(255,0,0),2)

        img = img[y-20:y+h-40, x-20:x+w+20] # for cropping

    cv_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return cv_rgb

# img = mpimg.imread(test_df['image'][1])

plt.figure(figsize=(5,5))

img = cv2.imread(PATH3)

c=face_detection(img)

plt.imshow(c)
c = transform.resize(c, (100,100))

plt.imshow(c)

# c = c / 255.0
predict_image = np.expand_dims(c, axis=0)
print(predict_image.shape)
print(predict_image)
a = model.predict(predict_image)
print(a)