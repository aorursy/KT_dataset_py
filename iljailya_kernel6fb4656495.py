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
import matplotlib.pyplot as plt
from PIL import Image
!unzip -q "/kaggle/input/hotdogornot/train.zip"
!unzip -q "/kaggle/input/hotdogornot/test.zip"
train_folder = "train_kaggle/"
print('Number of files in the train folder', len(os.listdir(train_folder)))
test_folder = "test_kaggle/"
print('Number of files in the test folder', len(os.listdir(test_folder)))
!mkdir train
!mkdir train/chili-dog
!mkdir train/hotdog
!mkdir train/frankfurter
!mkdir train/people
!mkdir train/pets
!mkdir train/food
!mkdir train/furniture
!mkdir validation
!mkdir validation/chili-dog
!mkdir validation/hotdog
!mkdir validation/frankfurter
!mkdir validation/people
!mkdir validation/pets
!mkdir validation/food
!mkdir validation/furniture
from shutil import copyfile
for dirname, _, filenames in os.walk('train_kaggle/'):
    for filename in filenames:
        if 'frankfurter' in filename:
            if np.random.uniform() >= 0.7:
                copyfile(os.path.join(dirname, filename), os.path.join('validation/frankfurter/', filename))
            else:
                copyfile(os.path.join(dirname, filename), os.path.join('train/frankfurter/', filename))
        elif 'chili-dog' in filename:
            if np.random.uniform() >= 0.7:
                copyfile(os.path.join(dirname, filename), os.path.join('validation/chili-dog/', filename))
            else:
                copyfile(os.path.join(dirname, filename), os.path.join('train/chili-dog/', filename))
        elif 'hotdog' in filename:
            if np.random.uniform() >= 0.7:
                copyfile(os.path.join(dirname, filename), os.path.join('validation/hotdog/', filename))
            else:
                copyfile(os.path.join(dirname, filename), os.path.join('train/hotdog/', filename))
        elif 'people' in filename:
            if np.random.uniform() >= 0.7:
                copyfile(os.path.join(dirname, filename), os.path.join('validation/people/', filename))
            else:
                copyfile(os.path.join(dirname, filename), os.path.join('train/people/', filename))
        elif 'pets' in filename:
            if np.random.uniform() >= 0.7:
                copyfile(os.path.join(dirname, filename), os.path.join('validation/pets/', filename))
            else:
                copyfile(os.path.join(dirname, filename), os.path.join('train/pets/', filename))
        elif 'food' in filename:
            if np.random.uniform() >= 0.7:
                copyfile(os.path.join(dirname, filename), os.path.join('validation/food/', filename))
            else:
                copyfile(os.path.join(dirname, filename), os.path.join('train/food/', filename))
        elif 'furniture' in filename:
            if np.random.uniform() >= 0.7:
                copyfile(os.path.join(dirname, filename), os.path.join('validation/furniture/', filename))
            else:
                copyfile(os.path.join(dirname, filename), os.path.join('train/furniture/', filename))
import tensorflow as tf
print(tf.__version__)
import tensorflow.keras as keras
mobile = keras.applications.MobileNetV2(include_top=True, weights='imagenet')
input_shape = mobile.layers[0].output_shape[0][1:3]
input_shape
from imgaug import augmenters as iaa
aug1 = iaa.GaussianBlur(sigma=(0, 1))
aug2 = iaa.AdditiveGaussianNoise(0, 0.05)

def additional_augmenation(image):
    if np.random.uniform() > 0.5:
        image = aug1.augment_image(image)
    if np.random.uniform() > 0.5:
        image = aug2.augment_image(image)
    return image
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

image_generator_train = ImageDataGenerator(rescale=1/255, rotation_range=10,
	zoom_range=0.1,
	width_shift_range=0.2,
	height_shift_range=0.2,
	shear_range=0.15,
	horizontal_flip=True,
	fill_mode="nearest", preprocessing_function=additional_augmenation)

image_generator_validation = ImageDataGenerator(rescale=1/255)



train_generator = image_generator_train.flow_from_directory(
    'train/',
    target_size=input_shape,
    batch_size=32,)

validation_generator = image_generator_validation.flow_from_directory(
    'validation/',
    target_size=input_shape,
    batch_size=32)
transfer_layer = mobile.layers[-2]
from tensorflow.python.keras.models import Model

feature_extractor = Model(inputs=mobile.input,
                   outputs=transfer_layer.output)
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.keras.layers import Dropout

model = Sequential()
model.add(feature_extractor)
model.add(Dense(1024,activation='relu'))
model.add(Dense(512,activation='elu'))
model.add(Dense(256,activation='elu'))
model.add(Dropout(0.3))
model.add(Dense(256,activation='relu'))
model.add(Dense(64,activation='elu'))
model.add(Dense(7, activation='softmax'))
def print_layer_trainable():
    for layer in feature_extractor.layers:
        print("{0}:\t{1}".format(layer.trainable, layer.name))
feature_extractor.trainable = True  
for layer in feature_extractor.layers:
    layer.trainable = False
print_layer_trainable( )
model.compile(
      optimizer=tf.keras.optimizers.Adam(),
      loss='categorical_crossentropy',
      metrics=['acc']
)
model.summary()
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint
checkpoint = ModelCheckpoint("best_weights.h5", monitor='val_acc', save_best_only=True, mode='max')
steps_per_epoch = np.ceil(train_generator.samples/train_generator.batch_size)
model.fit(train_generator, epochs=16, 
            steps_per_epoch = steps_per_epoch, validation_data = validation_generator, 
    validation_steps = validation_generator.samples // validation_generator.batch_size, callbacks=[checkpoint])
from tensorflow.keras.models import load_model
!mkdir test
!mkdir test/test
for dirname, _, filenames in os.walk('test_kaggle/'):
    for filename in filenames:
        copyfile(os.path.join(dirname, filename), os.path.join('test/test/', filename))
img_generator = ImageDataGenerator(rescale=1/255)
test_generator = img_generator.flow_from_directory(
    'test/',
    target_size=input_shape,
    batch_size=1150, shuffle=False)
bm = load_model("best_weights.h5")
res = bm.predict(test_generator)
classes = list(train_generator.class_indices.keys())
classes
pred = [classes[i] for i in np.argmax(res, axis=1)]
pred[:10]
labels_batch = list(zip(np.max(res, axis=1), pred))
labels_batch[:10]
for image_batch, label_batch in test_generator:
    print("Image batch shape: ", image_batch.shape)
    print("Labe batch shape: ", label_batch.shape)
    break
cols = 10
rows = 10
fig = plt.figure(figsize=(4 * cols - 1, 5 * rows - 1))

k = 0 
for i in range(cols):
    for j in range(rows):
        ax = fig.add_subplot(rows, cols, i * rows + j + 1)
        ax.grid('off')
        ax.axis('off')
        ax.imshow(image_batch[k])
        
        ax.set_title("{0:>6.2%} : {1}".format(labels_batch[k][0], labels_batch[k][1]), size=14)
        k += 1
plt.show()
data = pd.DataFrame(labels_batch)
data[1] = data[1].apply(lambda x: 1 if x == 'hotdog' or x == 'chili-dog' or x == 'frankfurter' else 0)
data[1].value_counts()
sample_submission = pd.read_csv("/kaggle/input/hotdogornot/sample_submission.csv")
sample_submission['label'] = data[1].values
sample_submission




