import os
import tqdm
import matplotlib.pyplot as plt
from keras import preprocessing, layers, models, optimizers
import numpy as np
#FAST_RUN = True # controls whether to run kernel fast
FAST_RUN = False
!ls ../input/test_set/test_set
!ls ../input/test_set/test_set/cats | wc -l
!ls ../input/test_set/test_set/dogs | wc -l
# there are about 2000 samples in the test set
!ls ../input/training_set/training_set/
!ls ../input/training_set/training_set/dogs | wc -l
!ls ../input/training_set/training_set/cats | wc -l
# there are about 8000 samples in the trianing set
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu',
                       input_shape=(150, 150, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy',
             optimizer=optimizers.RMSprop(lr=1e-4),
             metrics=['acc'])
model.summary()
path_cats = []
train_path_cats = '../input/training_set/training_set/cats'
for path in os.listdir(train_path_cats):
    if '.jpg' in path:
        path_cats.append(os.path.join(train_path_cats, path))
path_dogs = []
train_path_dogs = '../input/training_set/training_set/dogs'
for path in os.listdir(train_path_dogs):
    if '.jpg' in path:
        path_dogs.append(os.path.join(train_path_dogs, path))
len(path_dogs), len(path_cats)
# load training set
training_set = np.zeros((6000, 150, 150, 3), dtype='float32')
for i in range(6000):
    if i < 3000:
        path = path_dogs[i]
        img = preprocessing.image.load_img(path, target_size=(150, 150))
        training_set[i] = preprocessing.image.img_to_array(img)
    else:
        path = path_cats[i - 3000]
        img = preprocessing.image.load_img(path, target_size=(150, 150))
        training_set[i] = preprocessing.image.img_to_array(img)
training_set.shape
# load validation set
validation_set = np.zeros((2000, 150, 150, 3), dtype='float32')
for i in range(2000):
    if i < 1000:
        path = path_dogs[i + 3000]
        img = preprocessing.image.load_img(path, target_size=(150, 150))
        validation_set[i] = preprocessing.image.img_to_array(img)
    else:
        path = path_cats[i + 2000]
        img = preprocessing.image.load_img(path, target_size=(150, 150))
        validation_set[i] = preprocessing.image.img_to_array(img)
validation_set.shape
# make target tensor
train_labels = np.zeros((3000,))
train_labels = np.concatenate((train_labels, np.ones((3000,))))
validation_labels = np.zeros((1000,))
validation_labels = np.concatenate((validation_labels, np.ones((1000,))))
train_datagen = preprocessing.image.ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow(
    training_set,
    train_labels,
    batch_size=32)
validation_generator = train_datagen.flow(
    validation_set,
    validation_labels,
    batch_size=32)
# when augmenting data, you need to specify the step_per_epoch
# usually, (num_samples / batch_size) * 2.5
history = model.fit_generator(
    train_generator,
    steps_per_epoch=100,
    epochs= 3 if FAST_RUN else 30,
    validation_steps=50,
    validation_data=validation_generator,
)
model.save('cats_and_dogs_small_1.h5')
# plot error curves
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()
# we're overfitting when train and validation diverge
# Demo data augmentation
datagen = preprocessing.image.ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
)
# visualize data augmentations
plt.clf()
fnames = [os.path.join(train_path_cats, fname) for fname in os.listdir(train_path_cats)]
img_path = fnames[3]

img = preprocessing.image.load_img(img_path, target_size=(150, 150))
x = preprocessing.image.img_to_array(img)
x = x.reshape((1,) + x.shape)
plt.figure(figsize=(20,20))

i = 0
for batch in datagen.flow(x, batch_size=1):
    plt.subplot(2, 2, i + 1)
    plt.imshow(preprocessing.image.array_to_img(batch[0]))
    i += 1
    if i % 4 == 0:
        break
plt.show()
# Add a dropout layer to fight overfitting as well
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu',
                       input_shape=(150, 150, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dropout(0.5))                       # DROPOUT
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy',
             optimizer=optimizers.RMSprop(lr=1e-4),
             metrics=['acc'])
 # Use data augmentation
train_datagen = preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
)
train_generator = train_datagen.flow(
    training_set,
    train_labels,
    batch_size=32)

# do not augment validation data
test_datagen = preprocessing.image.ImageDataGenerator(rescale=1./255)
validation_generator = test_datagen.flow(
    validation_set,
    validation_labels,
    batch_size=32)
# train
history = model.fit_generator(
    train_generator,
    steps_per_epoch=100,
    epochs= 3 if FAST_RUN else 200, # use more epochs if you are not limited by 1 hour limit
    validation_data=validation_generator,
    validation_steps=50)
model.save('cats_and_dogs_small_2.h5')
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()
# our accuracy keeps going up
# you are overfitting when validation accuracy goes down again

# our loss keeps going down
# you are overfitting when validation loss increases again