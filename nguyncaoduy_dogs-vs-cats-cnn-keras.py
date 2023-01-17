# Basic libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



import os, zipfile, random



# Import the backend

import tensorflow as tf



# Data preprocessing

from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img

from tensorflow.keras.preprocessing import image

from tensorflow.keras.utils import to_categorical



# Model architecture

from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization

from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping, ReduceLROnPlateau



# Model evaluation

from sklearn.metrics import accuracy_score
# Defining constants

img_size = (150, 150)

input_dim = (150, 150, 3)

epochs = 50

batch_size = 20
# Load the dataset

local_zip = '/kaggle/input/dogs-vs-cats/train.zip'

zip_ref = zipfile.ZipFile(local_zip, 'r')

zip_ref.extractall('/kaggle/working')

zip_ref.close()
# Prepare training data

filenames = os.listdir("/kaggle/working/train")

categories = []

for filename in filenames:

    category = filename.split('.')[0]

    if category == 'dog':

        categories.append(1)

    else:

        categories.append(0)



df = pd.DataFrame({

    'filename': filenames,

    'category': categories

})
df.head()
df['category'].value_counts().plot.bar()
# See sample image

sample = random.choice(filenames)

image = load_img("/kaggle/working/train/"+sample)

plt.imshow(image)
# Convert numbers into strings

df["category"] = df["category"].replace({0: 'cat', 1: 'dog'})
# Divide the dataset into training and validation

train_df, validate_df = train_test_split(df, test_size=0.2, random_state=0)

train_df = train_df.reset_index(drop=True)

validate_df = validate_df.reset_index(drop=True)
# Model architecture (normal)

model = Sequential([

    Conv2D(32, (3, 3), padding = 'same', activation = 'relu', input_shape = input_dim),

    BatchNormalization(),

    Conv2D(32, (3, 3), activation = 'relu'),

    BatchNormalization(),

    MaxPool2D(2, 2),

    Dropout(0.2),

    Conv2D(64, (3, 3), padding = 'same', activation = 'relu'),

    BatchNormalization(),

    Conv2D(64, (3, 3), activation = 'relu'),

    BatchNormalization(),

    MaxPool2D(2, 2),

    Dropout(0.3),

    Conv2D(128, (3, 3), activation = 'relu'),

    BatchNormalization(),

    Conv2D(128, (3, 3), activation = 'relu'),

    BatchNormalization(),

    MaxPool2D(2, 2),

    Dropout(0.4),

    Flatten(),

    Dense(256, activation = 'relu'),

    Dropout(0.25),

    Dense(1, activation = 'sigmoid')

])
# Reduce LR

learning_rate_reduction = ReduceLROnPlateau(monitor = 'val_acc', 

                                            patience = 3,

                                            verbose = 1, 

                                            factor = 0.5, 

                                            min_lr = 0.00001)
# Early Stopping

early_stopping = EarlyStopping(monitor = 'val_acc',

                               patience = 10,

                               verbose = 1,

                               mode = 'auto',

                               restore_best_weights = True)
# Data Augmentation (only on training data)

train_datagen = ImageDataGenerator(rescale = 1./255,

                                   rotation_range = 40,  

                                   zoom_range = 0.2, 

                                   shear_range = 0.2,

                                   width_shift_range = 0.2, 

                                   height_shift_range = 0.2,  

                                   horizontal_flip = False, 

                                   vertical_flip = False)

train_generator = train_datagen.flow_from_dataframe(train_df, 

                                                    "/kaggle/working/train", 

                                                    x_col = 'filename',

                                                    y_col = 'category',

                                                    target_size = img_size,

                                                    class_mode = 'binary',

                                                    batch_size = batch_size)



validation_datagen = ImageDataGenerator(rescale = 1./255)

validation_generator = validation_datagen.flow_from_dataframe(validate_df, 

                                                              "/kaggle/working/train", 

                                                              x_col = 'filename',

                                                              y_col = 'category',

                                                              target_size = img_size,

                                                              class_mode = 'binary',

                                                              batch_size = batch_size)
# Compile the model

model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['acc'])
# Model summary

model.summary()
# Training

history = model.fit_generator(train_generator,

                              steps_per_epoch = train_df.shape[0] // batch_size,  

                              epochs = epochs,

                              validation_data = validation_generator,

                              validation_steps = validate_df.shape[0] // batch_size, 

                              verbose = 1,

                              callbacks = [learning_rate_reduction, early_stopping])
# Plot the learning curve

import matplotlib.pyplot as plt

acc = history.history['acc']

val_acc = history.history['val_acc']

loss = history.history['loss']

val_loss = history.history['val_loss']



epochs = range(len(acc))



plt.plot(epochs, acc, 'bo', label='Training accuracy')

plt.plot(epochs, val_acc, 'b', label='Validation accuracy')

plt.title('Training and validation accuracy')



plt.figure()



plt.plot(epochs, loss, 'bo', label='Training Loss')

plt.plot(epochs, val_loss, 'b', label='Validation Loss')

plt.title('Training and validation loss')

plt.legend()



plt.show()
# Load the test set

local_zip = '/kaggle/input/dogs-vs-cats/test1.zip'

zip_ref = zipfile.ZipFile(local_zip, 'r')

zip_ref.extractall('/kaggle/working')

zip_ref.close()
# Prepare Testing Data

test_filenames = os.listdir("/kaggle/working/test1")

test_df = pd.DataFrame({'filename': test_filenames})

nb_samples = test_df.shape[0]
test_df.head()
# Create Testing Generator

test_gen = ImageDataGenerator(rescale=1./255)

test_generator = test_gen.flow_from_dataframe(test_df, 

                                              "/kaggle/working/test1", 

                                              x_col = 'filename',

                                              y_col = None,

                                              class_mode = None,

                                              target_size = img_size,

                                              batch_size = batch_size,

                                              shuffle = False)
# Predict the result

predict = model.predict_generator(test_generator, steps = np.ceil(nb_samples/batch_size))



test_df['category'] = np.argmax(predict, axis=-1)



label_map = dict((v,k) for k,v in train_generator.class_indices.items())

test_df['category'] = test_df['category'].replace(label_map)



test_df['category'] = test_df['category'].replace({ 'dog': 1, 'cat': 0 })
# Submission

submission_df = test_df.copy()

submission_df['id'] = submission_df['filename'].str.split('.').str[0]

submission_df['label'] = submission_df['category']

submission_df.drop(['filename', 'category'], axis=1, inplace=True)

submission_df.to_csv('submission.csv', index=False)