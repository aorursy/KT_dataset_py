import tensorflow as tf

from tensorflow.keras import layers

from tensorflow.keras import models

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from tensorflow.keras.applications.vgg16 import VGG16
import cv2

import matplotlib.pyplot as plt
from albumentations import (

    Compose, RandomBrightness, JpegCompression, HueSaturationValue, RandomContrast, HorizontalFlip, Rotate, GaussianBlur, Cutout

)
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
tf.random.set_seed(0)
df = pd.read_csv('../input/lego-minifigures-classification/index.csv')

df.describe()
df.head()
train_set = df[df['train-valid'] == 'train']

valid_set = df[df['train-valid'] == 'valid']
train_set
valid_set
train_paths = []

for path in train_set['path'].values:

    train_paths.append(os.path.join('../input/lego-minifigures-classification/',path))

train_paths
valid_paths = []

for path in valid_set['path'].values:

    valid_paths.append(os.path.join('../input/lego-minifigures-classification/',path))

valid_paths
train_labels = train_set['class_id'].values

train_labels
valid_labels = valid_set['class_id'].values

valid_labels
dfmeta = pd.read_csv('../input/lego-minifigures-classification/metadata.csv')

no_of_classes = dfmeta.shape[0]

no_of_classes
class DataGenerator(tf.keras.utils.Sequence):

    

    def __init__(self, paths, labels = None, image_size = (512,512), batch_size = 32, num_classes = None, shuffle = False, transforms = False):

        self.paths = paths

        self.labels = labels

        self.image_size = image_size

        self.batch_size = batch_size

        self.num_classes = num_classes

        self.shuffle = shuffle

        self.transforms = transforms

        self.on_epoch_end()

        

    def __len__(self):

        return len(self.paths)//self.batch_size

    

    def __getitem__(self, index):

        indices = self.indices[index * self.batch_size : (index + 1) * self.batch_size]

        X, y = self.__get_data(indices)

        return X, y

    

    def __get_data(self, indices):

        batch = [self.paths[k] for k in indices]

        images = []

        for i in range(self.batch_size):

            img = cv2.imread(batch[i])

            img = cv2.resize(img, self.image_size)

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            if self.transforms:

                img = self.transforms(image = img)['image']

            img = img/255.0

            images.append(img)

        labels = [self.labels[k] - 1 for k in indices]

        return np.array(images), np.array(labels)       

    

    # this function is called at the end of every epoch

    def on_epoch_end(self):

        self.indices = np.arange(len(self.paths))

        if self.shuffle:

            np.random.shuffle(self.indices)

            

# function to carry out image augmentation

def transforms():

    return Compose([

                    Rotate(limit=40),

                    HorizontalFlip(p=0.5),

                    RandomBrightness(limit=0.2,p=0.5),

                    RandomContrast(limit=0.2, p=0.5),

                    JpegCompression(quality_lower=85, quality_upper=100, p=0.5),

                    HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),

                    GaussianBlur(blur_limit=(3, 7), always_apply=False, p=0.5),

                    Cutout(num_holes=8, max_h_size=8, max_w_size=8, fill_value=0, always_apply=False, p=0.5)

                    ])

    
IMAGE_SIZE = (512,512)
train_generator = DataGenerator(train_paths,

                               train_labels,

                               image_size = IMAGE_SIZE,

                               batch_size = 4,

                               num_classes = no_of_classes,

                               shuffle = True,

                               transforms = transforms())
valid_generator = DataGenerator(valid_paths,

                               valid_labels,

                               image_size = IMAGE_SIZE,

                               batch_size = 1,

                               num_classes = no_of_classes,

                               shuffle = False)
plt.figure(figsize = (16,16))

for row in range(4):

    images, labels = train_generator[row]

    for col in range(4):

        plt.subplot(4,4,(row * 4 + col) + 1)

        plt.imshow(images[col])

        plt.title(labels[col])
plt.figure(figsize = (16,16))

for i in range(16):

    image, label = valid_generator[i]

    plt.subplot(4,4,i + 1)

    plt.imshow(image[0])

    plt.title(label[0])
def create_model(input_shape):

    # initialize the base model as VGG16 model with input shape as (512,512,3)

    base_model = VGG16(input_shape = input_shape,

                       include_top = False,

                       weights = 'imagenet')



    # we do not have to train all of the layers

    for layer in base_model.layers:

        layer.trainable = False

        

    x = layers.Flatten()(base_model.output)

    x = layers.Dense(512, activation = 'relu')(x)

    x = layers.Dropout(0.5)(x)

    x = layers.Dense(no_of_classes, activation = 'softmax')(x)

    

    return models.Model(base_model.input,x)
model = create_model((512,512,3))
model.compile(loss = 'sparse_categorical_crossentropy',

             optimizer = Adam(learning_rate=0.0001),

             metrics = ['accuracy'])
# Stop training when the validation loss metric has stopped decreasing for 5 epochs.

early_stopping = EarlyStopping(monitor = 'val_loss',

                               patience = 5,

                               mode = 'min',

                               restore_best_weights = True)
# Save the model with the minimum validation loss

checkpoint = ModelCheckpoint('best_model.hdf5', 

                             monitor = 'val_loss',

                             verbose = 1,

                             mode = 'min', 

                             save_best_only = True)
EPOCHS = 50
history = model.fit(train_generator,

                    validation_data = valid_generator,

                    epochs = EPOCHS,

                    steps_per_epoch = len(train_generator),

                    validation_steps = len(valid_generator),

                    callbacks = [early_stopping, checkpoint])
model.summary()
acc = history.history['accuracy']

val_acc = history.history['val_accuracy']

loss = history.history['loss']

val_loss = history.history['val_loss']



epochs = range(len(acc))



plt.plot(epochs, acc, 'c-', label='Training accuracy')

plt.plot(epochs, val_acc, 'y-', label='Validation accuracy')

plt.title('Training and validation accuracy')

plt.legend()



plt.figure()



plt.plot(epochs, loss, 'c-', label='Training Loss')

plt.plot(epochs, val_loss, 'y-', label='Validation Loss')

plt.title('Training and validation loss')

plt.legend()



plt.show()
# load the best saved model as a new model

new_model = models.load_model('best_model.hdf5')



# Check its architecture

new_model.summary()
# Evaluate the restored model



actual_y = []

pred_y = []



for image, label in valid_generator:

    pred_y.extend(new_model.predict(image).argmax(axis = 1))

    actual_y.extend(label)
from sklearn.metrics import accuracy_score

acc = accuracy_score(actual_y, pred_y)

print('Accuracy: {:5.2f}%'.format(100*acc))