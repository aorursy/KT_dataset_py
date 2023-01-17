import os



import numpy as np

import pandas as pd

import cv2

import matplotlib.pyplot as plt



from tqdm import tqdm, tqdm_notebook



from keras import models, layers

from keras.layers import Conv2D, BatchNormalization, MaxPooling2D, Flatten, Dense, LeakyReLU, Dropout

from keras.applications import VGG16, densenet

from keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from keras.optimizers import Adam

from keras.preprocessing.image import ImageDataGenerator
base_dir = "../input/"

print(os.listdir(base_dir))



train_dir = os.path.join(base_dir, "train/train/")

test_dir = os.path.join(base_dir, "test/test/")



df_train = pd.read_csv(os.path.join(base_dir, "train.csv"))

print(df_train.head())
im = cv2.imread(train_dir + df_train["id"][0])

plt.imshow(im)
%load_ext tensorboard.notebook

%tensorboard --logdir logs
df_train['has_cactus'] = df_train['has_cactus'].astype(str)



batch_size = 64

train_size = 15750

validation_size = 1750



datagen = ImageDataGenerator(

    rescale=1./255,

    horizontal_flip=True,

    vertical_flip=True,

    validation_split=0.1)



data_args = {

    "dataframe": df_train,

    "directory": train_dir,

    "x_col": 'id',

    "y_col": 'has_cactus',

    "shuffle": True,

    "target_size": (32, 32),

    "batch_size": batch_size,

    "class_mode": 'binary'

}



train_generator = datagen.flow_from_dataframe(**data_args, subset='training')

validation_generator = datagen.flow_from_dataframe(**data_args, subset='validation')
ckpt_path = 'aerial_cactus_detection.hdf5'



tensorboard_cb = TensorBoard()

earlystop_cb = EarlyStopping(monitor='val_acc', patience=10, verbose=1, restore_best_weights=True)

reducelr_cb = ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=5, verbose=1)

modelckpt_cb = ModelCheckpoint(ckpt_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')



callbacks = [tensorboard_cb, earlystop_cb, reducelr_cb, modelckpt_cb]
model = models.Sequential([

    Conv2D(32, (3,3), input_shape=(32, 32, 3)),

    LeakyReLU(alpha=0.3),

    BatchNormalization(),

    Conv2D(32, (3,3)),

    LeakyReLU(alpha=0.3),

    BatchNormalization(),

    Conv2D(32, (3,3)),

    LeakyReLU(alpha=0.3),

    BatchNormalization(),

    MaxPooling2D(2,2),

   

    Conv2D(64, (3,3)),

    LeakyReLU(alpha=0.3),

    BatchNormalization(),

    Conv2D(64, (3,3)),

    LeakyReLU(alpha=0.3),

    BatchNormalization(),

    Conv2D(64, (3,3)),

    LeakyReLU(alpha=0.3),

    BatchNormalization(),

    MaxPooling2D(2,2),

    

    Flatten(),

    Dense(units=128),

    LeakyReLU(alpha=0.3),

    Dropout(0.4),

    Dense(units=64),

    LeakyReLU(alpha=0.3),

    Dropout(0.4),

    

    Dense(units=1, activation='sigmoid')

])



model.compile(loss='binary_crossentropy',

              optimizer=Adam(lr=0.001),

              metrics=['acc'])
history = model.fit_generator(train_generator,

              validation_data=validation_generator,

              steps_per_epoch=train_size//batch_size,

              validation_steps=validation_size//batch_size,

              epochs=100,

              shuffle=True,

              callbacks=callbacks, 

              verbose=2)
# Training plots

epochs = [i for i in range(1, len(history.history['loss'])+1)]



plt.plot(epochs, history.history['loss'], color='blue', label="training_loss")

plt.plot(epochs, history.history['val_loss'], color='red', label="validation_loss")

plt.legend(loc='best')

plt.title('loss')

plt.xlabel('epoch')

plt.show()



plt.plot(epochs, history.history['acc'], color='blue', label="training_accuracy")

plt.plot(epochs, history.history['val_acc'], color='red',label="validation_accuracy")

plt.legend(loc='best')

plt.title('accuracy')

plt.xlabel('epoch')

plt.show()
df_test = pd.read_csv(os.path.join(base_dir, "sample_submission.csv"))

print(df_test.head())

test_images = []

images = df_test['id'].values



for image_id in images:

    test_images.append(cv2.imread(os.path.join(test_dir, image_id)))

    

test_images = np.asarray(test_images)

test_images = test_images / 255.0

print(len(test_images))
pred = model.predict(test_images)

df_test['has_cactus'] = pred

df_test.to_csv('aerial-cactus-submission.csv', index = False)