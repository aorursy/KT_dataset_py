import numpy as np

import pandas as pd

from IPython.display import display

from tensorflow.keras.preprocessing.image import load_img, img_to_array

from tensorflow.keras.applications.vgg16 import VGG16

from tensorflow.keras.applications.vgg16 import preprocess_input

from tensorflow.keras import models, layers

from tensorflow.keras.optimizers import Adam

import matplotlib.pyplot as plt
labels = pd.read_csv('../input/labels.csv')

# sample_submission = pd.read_csv('../input/sample_submission.csv')
# display image

IMG = labels.iloc[8]

img_path = '../input/train/{}.jpg'.format(IMG[0])

display(load_img(img_path))

print(IMG[1])
# create index for subset data

np.random.seed(123)

index = np.array(labels.index)

np.random.shuffle(index)

train_index = index[:6400]

val_index = index[6400:9600]
# create lists with path to file and associated labels

def split_dataset(labels, index):

    x = []

    y = []

    for i, breed in labels.iloc[index].values:

        x.append('../input/train/{}.jpg'.format(i))

        y.append(breed)

    return x, y



x_train, y_train = split_dataset(labels, train_index)

x_val, y_val = split_dataset(labels, val_index)
# generator

from tensorflow.keras.utils import Sequence

# Here, `x_set` is list of path to the images

# and `y_set` are the associated classes.



class dog_sequence(Sequence):



    def __init__(self, x_set, y_set, batch_size):

        self.x = x_set

        self.y = pd.get_dummies(y_set)

        self.batch_size = batch_size



    def __len__(self):

        return int(np.ceil(len(self.x) / float(self.batch_size)))



    def __getitem__(self, idx):

        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]

        batch_y = self.y.iloc[idx * self.batch_size:(idx + 1) * self.batch_size]

        

        x = np.empty(shape=(128, 224, 224, 3))

        n = 0

        for file_name in batch_x:

            img = img_to_array(load_img(file_name, target_size=(224, 224)))

            img = np.expand_dims(img, 0)

            x[n,:,:,:] = img

            n += 1

        x = preprocess_input(x)

               

        return x, batch_y
# load VGG16 model

base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

base_model.trainable = False

# base_model.summary()
model = models.Sequential()

model.add(base_model)

model.add(layers.GlobalAveragePooling2D())

model.add(layers.Dense(512, activation='relu'))

model.add(layers.Dropout(0.4))

model.add(layers.Dense(256, activation='relu'))

model.add(layers.Dropout(0.4))

model.add(layers.Dense(120, activation='softmax'))



model.compile(loss='categorical_crossentropy',

              optimizer=Adam(lr=0.0001),

              metrics=['accuracy'])

model.summary()
from tensorflow.keras.callbacks import ModelCheckpoint   

checkpointer = ModelCheckpoint(filepath='myModel.hdf5', verbose=1, save_best_only=True,

                              save_weights_only=True)
hist = model.fit_generator(dog_sequence(x_train, y_train, batch_size=128),

                           validation_data=dog_sequence(x_val, y_val, batch_size=128),

                           epochs=10, verbose=1, callbacks=[checkpointer])

text = ' max acc: {:.3f}\n max val_acc {:.3f}'.format(max(hist.history['acc']), max(hist.history['val_acc']))

print(text)
plt.plot(hist.history['acc'], label='train');

plt.plot(hist.history['val_acc'], label='val');

plt.legend();
base_model.trainable = True

base_model.summary()
for layer in base_model.layers:

    if layer.name.startswith('block5'):

        layer.trainable = True

    else:

        layer.trainable = False

    print(layer.name, layer.trainable)

base_model.summary()
# model.save_weights('myModel.hdf5')

# model.load_weights('myModel.hdf5', by_name=True) # by_name=False