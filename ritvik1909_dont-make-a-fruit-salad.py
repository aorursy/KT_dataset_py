import numpy as np
import pandas as pd

from keras.models import Sequential
from keras import applications
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, GlobalAveragePooling2D
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import plot_model

import matplotlib.pyplot as plt
import seaborn as sns

import os, random
DIR_TRAIN = '../input/fruits/fruits-360/Training'
DIR_TEST = '../input/fruits/fruits-360/Test'
target_classes = os.listdir(DIR_TRAIN)
num_classes = len(target_classes)
print('Number of target classes:', num_classes)
print(list(enumerate(target_classes)))
training_set_distribution = [len(os.listdir(os.path.join(DIR_TRAIN, DIR))) for DIR in os.listdir(DIR_TRAIN)]
testing_set_distribution = [len(os.listdir(os.path.join(DIR_TEST, DIR))) for DIR in os.listdir(DIR_TEST)]
images = [
    (os.path.join(DIR_TRAIN, DIR, random.choice(
        os.listdir(os.path.join(DIR_TRAIN, DIR))
    )), DIR) for DIR in random.choices(os.listdir(DIR_TRAIN), k=15)
]

fig, ax = plt.subplots(3, 5, figsize=(20, 12))
fig.suptitle('Sample Images', fontsize=18)

for i, img in enumerate(images):
    ax[i//5][i%5].imshow(plt.imread(img[0]))
    ax[i//5][i%5].set_title(img[1], fontsize=12)
sns.set_style('darkgrid')
fig, ax = plt.subplots(figsize=(30, 6))

ind = np.arange(len(target_classes))    # the x locations for the groups
width = 0.35       # the width of the bars: can also be len(x) sequence

p1 = ax.bar(ind, training_set_distribution, width)
p2 = ax.bar(ind, testing_set_distribution, width, bottom=training_set_distribution)

plt.ylabel('Images')
plt.title('Distribution', fontsize=18)
plt.xticks(ind, target_classes, rotation=90)
plt.yticks(np.arange(0, 81, 10))
plt.legend((p1[0], p2[0]), ('Training', 'Testing'))
image_size = (100, 100, 3)

datagen = ImageDataGenerator(
    rescale = 1./255, shear_range = 0.2, zoom_range = 0.2, horizontal_flip = True
)
training_set = datagen.flow_from_directory(
    '../input/fruits/fruits-360/Training', 
    target_size = image_size[:2],  batch_size = 32, class_mode = 'categorical', color_mode='rgb'
)
validation_set = datagen.flow_from_directory(
    '../input/fruits/fruits-360/Test',  
    target_size = image_size[:2], batch_size = 32, class_mode = 'categorical', color_mode='rgb'
)
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=7)
filepath = "model.h5"
ckpt = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
rlp = ReduceLROnPlateau(monitor='loss', patience=3, verbose=1)
def cnn(image_size, num_classes):
    classifier = Sequential()
    classifier.add(Conv2D(256, (5, 5), input_shape=image_size, activation='relu', padding='same'))
    classifier.add(MaxPooling2D(pool_size = (2, 2)))
    classifier.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    classifier.add(MaxPooling2D(pool_size = (2, 2)))
    classifier.add(Flatten())
    classifier.add(Dense(num_classes, activation = 'softmax'))
    classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
    return classifier

neuralnetwork_cnn = cnn(image_size, num_classes)
neuralnetwork_cnn.summary()
plot_model(neuralnetwork_cnn, show_shapes=True)
history = neuralnetwork_cnn.fit_generator(
    generator=training_set, validation_data=validation_set,
    callbacks=[es, ckpt, rlp], epochs = 10, 
)
fig, ax = plt.subplots(figsize=(20, 6))
pd.DataFrame(history.history).iloc[:, :-1].plot(ax=ax)
base_model = applications.MobileNetV2(input_shape=image_size, include_top=False, weights='imagenet')
base_model.trainable = True # True => Fine Tuning False=> Feature Extraction
global_average_layer = GlobalAveragePooling2D()
prediction_layer = Dense(num_classes, activation='softmax')

neuralnetwork_mobilenet = Sequential([
  base_model,
  global_average_layer,
  prediction_layer
])

neuralnetwork_mobilenet.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
neuralnetwork_mobilenet.summary()
plot_model(neuralnetwork_mobilenet, show_shapes=True, expand_nested=True)
history = neuralnetwork_mobilenet.fit_generator(
    generator=training_set, validation_data=validation_set,
    callbacks=[es, ckpt, rlp], epochs = 10, 
)
fig, ax = plt.subplots(figsize=(20, 6))
pd.DataFrame(history.history).iloc[:, :-1].plot(ax=ax)
base_model = applications.InceptionV3(input_shape=image_size, include_top=False, weights='imagenet')
base_model.trainable = False # True => Fine Tuning False=> Feature Extraction
global_average_layer = GlobalAveragePooling2D()
prediction_layer = Dense(num_classes, activation='softmax')

neuralnetwork_inception = Sequential([
  base_model,
  global_average_layer,
  prediction_layer
])

neuralnetwork_inception.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
neuralnetwork_inception.summary()
plot_model(neuralnetwork_inception, show_shapes=True, expand_nested=True)
history = neuralnetwork_inception.fit_generator(
    generator=training_set, validation_data=validation_set,
    callbacks=[es, ckpt, rlp], epochs = 10, 
)
fig, ax = plt.subplots(figsize=(20, 6))
pd.DataFrame(history.history).iloc[:, :-1].plot(ax=ax)