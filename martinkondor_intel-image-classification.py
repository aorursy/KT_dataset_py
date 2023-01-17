%matplotlib inline



import os





import numpy as np

import seaborn as sns

from PIL import Image

from matplotlib import pyplot as plt

from keras.preprocessing.image import ImageDataGenerator

from keras.applications import xception 

from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, LeakyReLU

from keras.models import Sequential

from keras.constraints import max_norm





sns.set()
labels = os.listdir('../input/seg_train/seg_train')

fig = plt.figure(figsize=(12, 7))



for i, label in enumerate(labels):

    img = np.random.choice(os.listdir('../input/seg_train/seg_train/' + label))

    img = Image.open('../input/seg_train/seg_train/' + label + '/' + img)

    

    plt.subplot(2, 3, i + 1)

    plt.imshow(img)

    plt.xticks(())

    plt.yticks(())

    plt.title(label.upper())

    

del labels, fig
img_shape = (150, 150,)

batch_size = 64



# Preprocessing with: img / 255

train_generator = ImageDataGenerator(rescale=1. / 255., validation_split=.07)  # Split 7% of train set as validation set

test_generator = ImageDataGenerator(rescale=1. / 255.)



# Creating data generators

train_data_gen = train_generator.flow_from_directory('../input/seg_train/seg_train', batch_size=batch_size, target_size=img_shape, seed=0, subset='training')

val_data_gen = train_generator.flow_from_directory('../input/seg_train/seg_train', batch_size=batch_size, target_size=img_shape, seed=0, subset='validation')

test_data_gen = test_generator.flow_from_directory('../input/seg_test/seg_test', batch_size=batch_size, target_size=img_shape, seed=0)
# Load the pretrained model

pretrained_model = xception.Xception(include_top=False, weights='imagenet', input_shape=(150, 150, 3,), pooling=None)



# Building the model

model = Sequential()

model.add(pretrained_model)

model.add(Flatten())

model.add(Dense(6, activation='softmax'))



# Set the pretrained model to be non trainable

model.layers[0].trainable = False



model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

model.summary()
h = model.fit_generator(train_data_gen, steps_per_epoch=100, validation_data=val_data_gen, validation_steps=20, epochs=6)
y = h.history['acc']

x = range(len(y))



plt.figure(figsize=(14, 7,))

plt.title('Accuracy')

plt.plot(x, y, label='Train accuracy');

try:

    plt.plot(x, h.history['val_acc'], label='Validation accuracy');

except KeyError:

    pass

plt.legend();



print('Accuracy on test set:', model.evaluate_generator(test_data_gen, verbose=1, steps=50)[1])

model.save('model.h5')