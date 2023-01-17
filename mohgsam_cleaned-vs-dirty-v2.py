# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import os
import numpy as np
import matplotlib.pyplot as plt
import zipfile
print(os.listdir("../input/platesv2"))

with zipfile.ZipFile('../input/platesv2/plates.zip', 'r') as zip_obj:
   # Extract all the contents of zip file in current directory
   zip_obj.extractall('../kaggle/working/')
    
print('After zip extraction:')
print(os.listdir("../kaggle/working/"))
PATH = '../kaggle/working/plates'
print(os.listdir("../kaggle/working/plates"))
print(os.listdir("../kaggle/working/plates/train"))
train_dir = os.path.join(PATH, 'train')
test_dir = os.path.join(PATH, 'test')
train_cleaned_dir = os.path.join(train_dir, 'cleaned')  
train_dirty_dir = os.path.join(train_dir, 'dirty')
train_cleaned_dir
print(os.listdir('../kaggle/working/plates/train/cleaned'))
print(len(os.listdir('../kaggle/working/plates/train/cleaned')))
num_cleaned_tr = len(os.listdir(train_cleaned_dir))
num_dirty_tr = len(os.listdir(train_dirty_dir))

num_test_val = len(os.listdir(test_dir))
#num_dirty_val = len(os.listdir(validation_dirty_dir))

total_train = num_cleaned_tr + num_dirty_tr
total_val = num_test_val
print(total_train)
print(total_val)
batch_size = 4
epochs = 50
IMG_HEIGHT = 150
IMG_WIDTH = 150
train_image_generator = ImageDataGenerator(rescale=1./255) 
validation_image_generator = ImageDataGenerator(rescale=1./255)
train_data_gen = train_image_generator.flow_from_directory(batch_size=batch_size,
                                                           directory=train_dir,
                                                           shuffle=True,
                                                           target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                           class_mode='binary')
val_data_gen = validation_image_generator.flow_from_directory(batch_size=batch_size,
                                                              directory= test_dir,
                                                              target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                              class_mode='binary')
sample_training_images, _ = next(train_data_gen)
def plotImages(images_arr):
    fig, axes = plt.subplots(1, 5, figsize=(20,20))
    axes = axes.flatten()
    for img, ax in zip( images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()
plotImages(sample_training_images[:5])
model = Sequential([
    Conv2D(16, 3, padding='same', activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH ,3)),
    MaxPooling2D(),
    Conv2D(32, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Conv2D(64, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(1)
])
model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])
model.summary()
history = model.fit(
    train_data_gen,
    steps_per_epoch = total_train // batch_size,
    epochs = epochs,
    
)
image_gen_train_ = ImageDataGenerator(
                    rescale=1./255,
                    rotation_range=45,
                    width_shift_range=.15,
                    height_shift_range=.15,
                    horizontal_flip=True,
                    zoom_range=0.5
                    )
train_data_gen_ = image_gen_train_.flow_from_directory(batch_size=batch_size,
                                                     directory=train_dir,
                                                     shuffle=True,
                                                     target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                     class_mode='binary')
test_dir
augmented_images = [train_data_gen[0][0][0] for i in range(5)]
plotImages(augmented_images)
model_new = Sequential([
    Conv2D(16, 3, padding='same', activation='relu', 
           input_shape=(IMG_HEIGHT, IMG_WIDTH ,3)),
    MaxPooling2D(),
    Dropout(0.2),
    Conv2D(32, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Conv2D(64, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Dropout(0.2),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(1)
])
model_new.compile(optimizer='adam',
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=['accuracy'])

model_new.summary()
history = model_new.fit(
    train_data_gen_,
    steps_per_epoch = total_train // batch_size,
    epochs = epochs,
    
)
image_gen_test = ImageDataGenerator()
test_data_gen = image_gen_test.flow_from_directory(  
        directory='../kaggle/working/plates/',
        classes=['test'],
        target_size = (IMG_HEIGHT, IMG_WIDTH),
        batch_size = 1,
        shuffle = False,        
        class_mode = None)
test_data_gen.reset()
predict = model.predict_generator(test_data_gen, steps = len(test_data_gen.filenames))
len(predict)
sub_df = pd.read_csv('../input/platesv2/sample_submission.csv')
sub_df.head()
sub_df['label'] = predict
sub_df['label'] = sub_df['label'].apply(lambda x: 'dirty' if x > 0.5 else 'cleaned')
sub_df.head()
sub_df['label'].value_counts()
sub_df.to_csv('sub.csv', index=False)
print('Done')