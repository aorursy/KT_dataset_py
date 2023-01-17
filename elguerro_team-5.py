import numpy as np

import pandas as pd

import keras

from keras import backend as k

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten

from keras.layers import Conv2D, MaxPooling2D, Activation

from keras.optimizers import SGD

from keras.preprocessing import image

from keras.preprocessing.image import ImageDataGenerator

from tqdm import tqdm



df_test = pd.read_csv('sample.csv', dtype={'Id': 'str', 'Category': 'int16'})



img_width, img_height = 512, 512



train_data_dir = 'Images_Aug/Train'

nb_train_samples = 2000

nb_validation_samples = 800

epochs = 11

batch_size = 16



if k.image_data_format() == 'channels_first':

    input_shape = (3, img_width, img_height)

else:

    input_shape = (img_width, img_height, 3)



model = Sequential()

model.add(Conv2D(48, (3, 3), input_shape=input_shape))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))



model.add(Conv2D(32, (3, 3)))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))



model.add(Conv2D(64, (3, 3)))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))



model.add(Flatten())

model.add(Dense(64))

model.add(Activation('relu'))

model.add(Dropout(0.5))

model.add(Dense(1))

model.add(Activation('sigmoid'))



model.compile(loss='binary_crossentropy',

              optimizer='rmsprop',

              metrics=['accuracy'])



datagen = ImageDataGenerator(validation_split=.2, rescale=1. / 255, rotation_range=90, horizontal_flip=True)



train_generator = datagen.flow_from_directory(

    train_data_dir,

    target_size=(img_width, img_height),

    batch_size=batch_size,

    class_mode='binary',

    subset='training')



validation_generator = datagen.flow_from_directory(

    train_data_dir,

    target_size=(img_width, img_height),

    batch_size=batch_size,

    class_mode='binary',

    subset='validation')



model.fit_generator(

    train_generator,

    steps_per_epoch=nb_train_samples // batch_size,

    epochs=epochs,

    validation_data=validation_generator,

    validation_steps=nb_validation_samples // batch_size)



print(model.summary())



test_image = []

for jpg in tqdm(range(df_test.shape[0])):

    img = image.load_img('Images/Testing/'+df_test['Id'][jpg], target_size=(512, 512, 3), color_mode='rgb')

    img = image.img_to_array(img)

    img = img / 255.0

    test_image.append(img)



x_pred = np.array(test_image)

y_pred = df_test['Category'].values



pred = model.predict(x_pred)

df_pred = pd.read_csv('sample.csv')

pred = np.rint(pred)

pred = pred.astype(int)

df_pred['Category'] = pred

df_pred.to_csv('prediction_Aug.csv', index=False)

model.save_weights('weights.csv')