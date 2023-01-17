#load libraries for data manipulation and visualization
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import os
import random
from tensorflow.keras.preprocessing.image import load_img
# warnings
import string
import warnings
warnings.filterwarnings('ignore')
!unzip -q /kaggle/input/dogs-vs-cats/train.zip
!unzip -q /kaggle/input/dogs-vs-cats/test1.zip
TRAIN_DIR = "/kaggle/working/train/"
TEST_DIR = "/kaggle/working/test1/"

# gather train data into a dataframe
filenames = os.listdir(TRAIN_DIR )
categories = []
for filename in filenames:
    category = filename.split('.')[0]
    if category == 'dog':
        categories.append('dog')
    else:
        categories.append('cat')

all_df = pd.DataFrame({
    'filename': filenames,
    'category': categories
})

# gather test data into a dataframe
test_filenames = os.listdir(TEST_DIR)
test_df = pd.DataFrame({
    'id': test_filenames
})
# display train data
all_df.sample(5)
# show counts for train data
all_df['category'].value_counts()
# display test data
test_df.sample(5)
# show sample size of test data
test_df.shape[0]
# display sample train images
sample = all_df.head(9)
sample.head()
plt.figure(figsize=(12, 12))
for index, row in sample.iterrows():
    filename = row['filename']
    category = row['category']
    img = load_img(TRAIN_DIR+filename, target_size=(128,128))
    plt.subplot(3, 3, index+1)
    plt.imshow(img)
    plt.xlabel(filename)
plt.tight_layout()
plt.show()
from sklearn.model_selection import train_test_split
# split into train/validate 
train_df, validate_df = train_test_split(all_df, test_size=0.20, random_state=0)
train_df = train_df.reset_index(drop=True)
validate_df = validate_df.reset_index(drop=True)
# show the count by category for train set
train_df['category'].value_counts()
# show the count by category for validate set
validate_df['category'].value_counts()
# define train data augmentation configuration
from tensorflow.keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(
        rescale=1./255,
rotation_range=30,
width_shift_range=0.15,
height_shift_range=0.15,
shear_range=0.15,
zoom_range=0.15,
horizontal_flip=True,
fill_mode='nearest')
# using ImageDataGenerator to generate sample images
sample_df = train_df.sample(n=1).reset_index(drop=True)
sample_generator = train_datagen.flow_from_dataframe(
    sample_df, 
    TRAIN_DIR, 
    x_col='filename',
    y_col='category',
    target_size = (128, 128),
    class_mode='categorical'
)
plt.figure(figsize=(8, 8))
for i in range(0, 4):
    plt.subplot(2, 2, i+1)
    for X, Y in sample_generator:
        image = X[0]
        plt.imshow(image)
        break
plt.tight_layout()
plt.show()
# reading train data
train_generator = train_datagen.flow_from_dataframe(
        train_df, 
        TRAIN_DIR,
        x_col='filename',
        y_col='category',
        target_size=(128, 128),
        batch_size=75,
        class_mode='binary')
# reading validation data
test_datagen = ImageDataGenerator(rescale=1./255)
validation_generator = test_datagen.flow_from_dataframe(
        validate_df, 
        TRAIN_DIR,
        x_col='filename',
        y_col='category',
        target_size=(128, 128),
        batch_size=50,
        class_mode='binary')
# reading test data
test_generator = test_datagen.flow_from_dataframe(
        test_df, 
        TEST_DIR,
        x_col='id',
        y_col=None,
        class_mode=None,
        target_size=(128, 128),
        batch_size=12500//50)
## define a custome cnn
from tensorflow.keras import layers
from tensorflow.keras import models
model = models.Sequential()
# convolutional-base
model.add(layers.Conv2D(32, (3, 3), activation='relu',
input_shape=(128, 128, 3)))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D((2, 2)))

# densely connected classifier
model.add(layers.Flatten())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.summary()
# optimizing model performance
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
callbacks = [
    EarlyStopping(patience=5, verbose=1),
    ReduceLROnPlateau(factor=0.25, patience=2, min_lr=0.00001, verbose=1),
    ModelCheckpoint('model2.h5', verbose=1, save_best_only=True, save_weights_only=True)
]
# configuring the model for training
import  tensorflow.keras.optimizers as optimizers
model.compile(loss='binary_crossentropy',
optimizer=optimizers.RMSprop(lr=.001),
metrics=['acc'])
%%time
# fitting the model using a batch generator
history = model.fit_generator(
        train_generator,
        steps_per_epoch=20000//75,
        epochs=30,
        validation_data=validation_generator,
        validation_steps=5000/50,
        callbacks=callbacks,
        use_multiprocessing=True,
        workers=4)
# displaying curves of loss and accuracy during training
import matplotlib.pyplot as plt
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'g', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'g', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()
# making predictions
#%%time
predictions1 = model.predict_generator(test_generator, steps=np.ceil(12500/50))
# extract id's of test set
submit_df = test_df.copy()
submit_df['id'] = submit_df['id'].str.split('.').str[0]

# converting predictions to 1 and 0
predictions1 = [1 if y > 0.5 else 0 for y in predictions1]

submit_df['label'] = predictions1

# restore back to class names (dog or cat)
label_map = dict((v,k) for k,v in train_generator.class_indices.items())
submit_df['label'] = submit_df['label'].replace(label_map)

# encoding according to submission format, dog = 1, cat = 0
submit_df['label'] = submit_df['label'].replace({ 'dog': 1, 'cat': 0 })

submit_df.to_csv('submission1.csv', index=False)
submit_df.sample(5)
# using the pretrained convolutional base
from tensorflow.keras.applications import VGG16
conv_base = VGG16(weights='imagenet',
include_top=False,
input_shape=(128, 128, 3))
conv_base.summary()
model = models.Sequential()
model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.summary()
# freezing all layers up to a specific one
conv_base.trainable = True
set_trainable = False
for layer in conv_base.layers:
    if layer.name == 'block5_conv1':
        set_trainable = True
    if set_trainable:
        layer.trainable = True
    else:
        layer.trainable = False

# optimizing model performance
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
callbacks = [
    EarlyStopping(patience=5, verbose=1),
    ReduceLROnPlateau(factor=0.25, patience=2, min_lr=0.000001, verbose=1),
    ModelCheckpoint('model3.h5', verbose=1, save_best_only=True, save_weights_only=True)
]
# configuring the model for training
model.compile(optimizer=optimizers.RMSprop(lr=1e-5),
loss='binary_crossentropy',
metrics=['acc'])

%%time
# fitting the model 
history = model.fit_generator(
        train_generator,
        steps_per_epoch=20000//75,
        epochs=30,
        validation_data=validation_generator,
        validation_steps=5000//50,
        callbacks=callbacks)
# plotting the results
import matplotlib.pyplot as plt
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'r', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'r', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()
# making predictions
#%%time
predictions2 = model.predict_generator(test_generator, steps=np.ceil(12500/50))
# extract id's of test set
submit_df = test_df.copy()
submit_df['id'] = submit_df['id'].str.split('.').str[0]

# converting predictions to 1 and 0
predictions2 = [1 if y > 0.5 else 0 for y in predictions2]

submit_df['label'] = predictions2

# restore back to class names (dog or cat)
label_map = dict((v,k) for k,v in train_generator.class_indices.items())
submit_df['label'] = submit_df['label'].replace(label_map)

# encoding according to submission format, dog = 1, cat = 0
submit_df['label'] = submit_df['label'].replace({ 'dog': 1, 'cat': 0 })

submit_df.to_csv('submission2.csv', index=False)
submit_df.sample(5)