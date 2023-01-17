import os

import numpy as np

import pandas as pd



import tensorflow as tf

from tensorflow.keras import layers



import matplotlib.pyplot as plt

from PIL import Image
input_dir = '/kaggle/input/'



# for root, dirs, files in os.walk(input_dir):

#     for f in files:

#         print(os.path.join(root, f))
input_train_file = os.path.join(input_dir, 'aerial-cactus-identification/train.csv')



train_raw_df = pd.read_csv(input_train_file)

train_raw_df
train_raw_df['has_cactus'] = train_raw_df['has_cactus'].astype(str)
val_count = int(len(train_raw_df) * 0.1)



val_df_0 = train_raw_df.loc[train_raw_df['has_cactus'] == '0'].sample(n=val_count // 2, random_state=13)

val_df_1 = train_raw_df.loc[train_raw_df['has_cactus'] == '1'].sample(n=val_count // 2, random_state=13)



val_df = pd.concat([val_df_0, val_df_1])



train_df = train_raw_df.drop(val_df.index)
model = tf.keras.Sequential()



model.add(layers.Conv2D(128, kernel_size=3, input_shape=(32, 32, 3), padding='same', activation="relu"))

model.add(layers.Conv2D(128, kernel_size=3, padding='same', activation="relu"))

model.add(layers.MaxPooling2D(2))



model.add(layers.Dropout(0.1))



model.add(layers.Conv2D(256, kernel_size=3, padding='same', activation="relu"))

model.add(layers.Conv2D(256, kernel_size=3, padding='same', activation="relu"))

model.add(layers.MaxPooling2D(2))



model.add(layers.Dropout(0.1))



model.add(layers.Conv2D(512, kernel_size=3, padding='same', activation="relu"))

model.add(layers.Conv2D(512, kernel_size=3, padding='same', activation="relu"))

model.add(layers.MaxPooling2D(2))



model.add(layers.Dropout(0.1))



model.add(layers.Flatten())



model.add(layers.Dense(2048, activation = "relu"))



model.add(layers.Dense(1, activation='sigmoid'))



model.compile(optimizer=tf.keras.optimizers.Adam(0.0001),

              loss=tf.keras.losses.BinaryCrossentropy(),

              metrics=[tf.keras.metrics.BinaryAccuracy()])
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(

    rescale=1./255,

    horizontal_flip=True,

    vertical_flip=True,

    rotation_range=90,

    width_shift_range=0.2,

    height_shift_range=0.2,

    shear_range=0.2,

    zoom_range=0.2)



val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(

    rescale=1./255)



train_images_dir = os.path.join(input_dir, 'aerial-cactus-identification/train/train')



train_generator = train_datagen.flow_from_dataframe(

    dataframe=train_df,

    directory=train_images_dir,

    x_col="id",

    y_col="has_cactus",

    target_size=(32, 32),

    batch_size=1024,

    class_mode='binary')



val_generator = val_datagen.flow_from_dataframe(

    dataframe=val_df,

    directory=train_images_dir,

    x_col="id",

    y_col="has_cactus",

    target_size=(32, 32),

    batch_size=1024,

    class_mode='binary')
model.fit_generator(

    train_generator,

    epochs=100,

    validation_data=val_generator)
test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(

    rescale=1./255)



test_images_dir = os.path.join(input_dir, 'aerial-cactus-identification/test/test')



test_generator = test_datagen.flow_from_directory(

    test_images_dir,

    target_size=(32, 32),

    batch_size=1024,

    shuffle=False,

    classes=[''],

    class_mode=None)
predict = model.predict_generator(test_generator)

predict
filenames = test_generator.filenames



output = pd.DataFrame({"id": filenames,

                       "has_cactus": (np.reshape(predict, -1) >= 0.5).astype(int)})

output
output.to_csv("submission.csv", index=False)