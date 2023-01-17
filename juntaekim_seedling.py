import os

from keras import layers

from keras import models

from keras.preprocessing.image import ImageDataGenerator

from keras import optimizers

import matplotlib.pyplot as plt

import numpy as np
base_dir ='../input/seedling-data-sets-from-marsh/new'
train_dir = os.path.join(base_dir, 'train')

val_dir = os.path.join(base_dir, 'validation')

test_dir = os.path.join(base_dir, 'test')
train_black_grass_dir = os.path.join(train_dir, 'black grass')

train_charlock_dir = os.path.join(train_dir, 'charlock')

train_cleavers_dir = os.path.join(train_dir, 'cleavers')

train_common_chikweed_dir = os.path.join(train_dir, 'common chikweed')

train_common_wheat_dir = os.path.join(train_dir, 'common wheat')

train_fat_hen_dir = os.path.join(train_dir, 'fat hen')

train_loose_silky_bent_chikweed_dir = os.path.join(train_dir, 'loose_silky_bent')

train_maize_dir = os.path.join(train_dir, 'maize')

train_scentless_mayweed_dir = os.path.join(train_dir, 'scentless mayweed')

train_shepherds_purse_dir = os.path.join(train_dir, 'shepherds purse')

train_small_flowered_cranesbill_dir = os.path.join(train_dir, 'small flowered cranesbill')

train_sugar_beet_dir = os.path.join(train_dir, 'sugar beet')
val_black_grass_dir = os.path.join(val_dir, 'black grass')

val_charlock_dir = os.path.join(val_dir, 'charlock')

val_cleavers_dir = os.path.join(val_dir, 'cleavers')

val_common_chikweed_dir = os.path.join(val_dir, 'common chikweed')

val_common_wheat_dir = os.path.join(val_dir, 'common wheat')

val_fat_hen_dir = os.path.join(val_dir, 'fat hen')

val_loose_silky_bent_chikweed_dir = os.path.join(val_dir, 'loose_silky_bent')

val_maize_dir = os.path.join(val_dir, 'maize')

val_scentless_mayweed_dir = os.path.join(val_dir, 'scentless mayweed')

val_shepherds_purse_dir = os.path.join(val_dir, 'shepherds purse')

val_small_flowered_cranesbill_dir = os.path.join(val_dir, 'small flowered cranesbill')

val_sugar_beet_dir = os.path.join(val_dir, 'sugar beet')
test_black_grass_dir = os.path.join(test_dir, 'black grass')

test_charlock_dir = os.path.join(test_dir, 'charlock')

test_cleavers_dir = os.path.join(test_dir, 'cleavers')

test_common_chikweed_dir = os.path.join(test_dir, 'common chikweed')

test_common_wheat_dir = os.path.join(test_dir, 'common wheat')

test_fat_hen_dir = os.path.join(test_dir, 'fat hen')

test_loose_silky_bent_chikweed_dir = os.path.join(test_dir, 'loose_silky_bent')

test_maize_dir = os.path.join(test_dir, 'maize')

test_scentless_mayweed_dir = os.path.join(test_dir, 'scentless mayweed')

test_shepherds_purse_dir = os.path.join(test_dir, 'shepherds purse')

test_small_flowered_cranesbill_dir = os.path.join(test_dir, 'small flowered cranesbill')

test_sugar_beet_dir = os.path.join(test_dir, 'sugar beet')
model = models.Sequential()

model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape= (161, 161,3)))

model.add(layers.Conv2D(32, (3,3), activation='relu'))

model.add(layers.MaxPooling2D(2,2))

model.add(layers.Conv2D(64, (3,3), activation='relu'))

model.add(layers.Conv2D(64, (3,3), activation='relu'))

model.add(layers.MaxPooling2D(2,2))

model.add(layers.Conv2D(128, (3,3), activation='relu'))

model.add(layers.Conv2D(128, (3,3), activation='relu'))

model.add(layers.Conv2D(128, (3,3), activation='relu'))

model.add(layers.MaxPooling2D(2,2))

model.add(layers.Flatten())

model.add(layers.Dropout(0.5))

model.add(layers.Dense(512, activation='relu'))

model.add(layers.Dense(12, activation='sigmoid'))

model.summary()
model.compile(

    loss= 'categorical_crossentropy',

    optimizer=optimizers.RMSprop(lr=1e-4),

    metrics=['acc']

)
train_datagen = ImageDataGenerator(rescale=1./255,

    rotation_range=70,

    width_shift_range=0.1,

    height_shift_range=0.1,

    shear_range=0.1,

    zoom_range=0.1,

    horizontal_flip=True,

    fill_mode='nearest')



test_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(

    train_dir,

    target_size=(161, 161),

    batch_size=40,

    class_mode='categorical'

)



validation_generator = test_datagen.flow_from_directory(

    val_dir,

    target_size=(161, 161),

    batch_size=40,

    class_mode='categorical'

)
history = model.fit_generator(

    train_generator,

    steps_per_epoch = 100,

    epochs = 50,

    validation_data = validation_generator,

    validation_steps = 50)
acc = history.history['acc']

val_acc =  history.history['val_acc']

loss = history.history['loss']

val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label = 'Training acc')

plt.plot(epochs, val_acc, 'b', label = 'Validation acc')

plt.title('Training and validation accuracy')

plt.legend()

plt.figure()
plt.plot(epochs, loss, 'bo', label = 'Training loss')

plt.plot(epochs, val_loss, 'b', label = 'Validation loss')

plt.title('Training and validation loss')

plt.legend()

plt.show()
test_generator = test_datagen.flow_from_directory(

    test_dir,

    target_size = (161, 161),

    batch_size= 40,

    class_mode = 'categorical'

)
test_loss, test_acc = model.evaluate_generator(test_generator, steps=80)

print('test acc:', test_acc)