import numpy as np

import pandas as pd

import os

import shutil
original_dataset_dir = '../input/dogs-vs-cats/train/train/'
print('total training images:', len(os.listdir(original_dataset_dir)))
base_dir = '/' # Current directory
# Directories for our training,

# validation and test splits

train_dir = os.path.join(base_dir, 'train')

os.mkdir(train_dir)



validation_dir = os.path.join(base_dir, 'validation')

os.mkdir(validation_dir)



test_dir = os.path.join(base_dir, 'test')

os.mkdir(test_dir)



# Directory with our training cat pictures

train_cats_dir = os.path.join(train_dir, 'cats')

os.mkdir(train_cats_dir)



# Directory with our training dog pictures

train_dogs_dir = os.path.join(train_dir, 'dogs')

os.mkdir(train_dogs_dir)



# Directory with our validation cat pictures

validation_cats_dir = os.path.join(validation_dir, 'cats')

os.mkdir(validation_cats_dir)



# Directory with our validation dog pictures

validation_dogs_dir = os.path.join(validation_dir, 'dogs')

os.mkdir(validation_dogs_dir)



# Directory with our validation cat pictures

test_cats_dir = os.path.join(test_dir, 'cats')

os.mkdir(test_cats_dir)



# Directory with our validation dog pictures

test_dogs_dir = os.path.join(test_dir, 'dogs')

os.mkdir(test_dogs_dir)



# Copy first 8000 cat images to train_cats_dir

fnames = ['cat.{}.jpg'.format(i) for i in range(8000)]

for fname in fnames:

    src = os.path.join(original_dataset_dir, fname)

    dst = os.path.join(train_cats_dir, fname)

    shutil.copyfile(src, dst)



# Copy next 2000 cat images to validation_cats_dir

fnames = ['cat.{}.jpg'.format(i) for i in range(8000, 10000)]

for fname in fnames:

    src = os.path.join(original_dataset_dir, fname)

    dst = os.path.join(validation_cats_dir, fname)

    shutil.copyfile(src, dst)

    

# Copy next 1000 cat images to test_cats_dir

fnames = ['cat.{}.jpg'.format(i) for i in range(10000, 11000)]

for fname in fnames:

    src = os.path.join(original_dataset_dir, fname)

    dst = os.path.join(test_cats_dir, fname)

    shutil.copyfile(src, dst)

    

# Copy first 8000 dog images to train_dogs_dir

fnames = ['dog.{}.jpg'.format(i) for i in range(8000)]

for fname in fnames:

    src = os.path.join(original_dataset_dir, fname)

    dst = os.path.join(train_dogs_dir, fname)

    shutil.copyfile(src, dst)

    

# Copy next 2000 dog images to validation_dogs_dir

fnames = ['dog.{}.jpg'.format(i) for i in range(8000, 10000)]

for fname in fnames:

    src = os.path.join(original_dataset_dir, fname)

    dst = os.path.join(validation_dogs_dir, fname)

    shutil.copyfile(src, dst)

    

# Copy next 1000 dog images to test_dogs_dir

fnames = ['dog.{}.jpg'.format(i) for i in range(10000, 11000)]

for fname in fnames:

    src = os.path.join(original_dataset_dir, fname)

    dst = os.path.join(test_dogs_dir, fname)

    shutil.copyfile(src, dst)
print('total training cat images:', len(os.listdir(train_cats_dir)))

print('total training dog images:', len(os.listdir(train_dogs_dir)))



print('\ntotal validation cat images:', len(os.listdir(validation_cats_dir)))

print('total validation dog images:', len(os.listdir(validation_dogs_dir)))



print('\ntotal test cat images:', len(os.listdir(test_cats_dir)))

print('total test dog images:', len(os.listdir(test_dogs_dir)))
from keras import layers

from keras import models



model = models.Sequential()

model.add(layers.Conv2D(64, (3, 3),

                        activation='relu',

                        input_shape=(150, 150, 3)))

model.add(layers.MaxPooling2D((2, 2)))



model.add(layers.Conv2D(32, (3, 3), activation='relu'))

model.add(layers.MaxPooling2D((2, 2)))



model.add(layers.Conv2D(32, (3, 3), activation='relu'))

model.add(layers.MaxPooling2D((2, 2)))



model.add(layers.Flatten())

model.add(layers.Dense(512, activation='relu'))

model.add(layers.Dense(1, activation='sigmoid'))
model.summary()
from keras import optimizers



model.compile(loss='binary_crossentropy',

              optimizer=optimizers.RMSprop(lr=0.0001),

              metrics=['acc'])
from keras.preprocessing.image import ImageDataGenerator



# All images will be rescaled by 1/255

train_datagen = ImageDataGenerator(rescale=1./255)

test_datagen = ImageDataGenerator(rescale=1./255)



train_generator = train_datagen.flow_from_directory(

        # This is the target directory

        train_dir,

        # All images will be resized to 150x150

        target_size=(150, 150),

        batch_size=20,

        # Since we use binary_crossentropy loss, we need binary labels

        class_mode='binary')



validation_generator = test_datagen.flow_from_directory(

        validation_dir,

        target_size=(150, 150),

        batch_size=20,

        class_mode='binary')
for data_batch, labels_batch in train_generator:

    print('data batch shape:', data_batch.shape)

    print('labels batch shape:', labels_batch.shape)

    break
history = model.fit_generator(

      train_generator,

      steps_per_epoch=100,

      epochs=100,

      validation_data=validation_generator,

      validation_steps=50)
model.save('CvD_Model_01.h5')
import matplotlib.pyplot as plt



acc = history.history['acc']

val_acc = history.history['val_acc']

loss = history.history['loss']

val_loss = history.history['val_loss']



epochs = range(len(acc))



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
datagen = ImageDataGenerator(

      rotation_range=40,

      width_shift_range=0.2,

      height_shift_range=0.2,

      shear_range=0.2,

      zoom_range=0.4,

      horizontal_flip=True,

      fill_mode='nearest')
# This is module with image preprocessing utilities

from keras.preprocessing import image



fnames = [os.path.join(train_cats_dir, fname) for fname in os.listdir(train_cats_dir)]



# We pick one image to "augment"

img_path = fnames[3]



# Read the image and resize it

img = image.load_img(img_path, target_size=(150, 150))



# Convert it to a Numpy array with shape (150, 150, 3)

x = image.img_to_array(img)



# Reshape it to (1, 150, 150, 3)

x = x.reshape((1,) + x.shape)



# The .flow() command below generates batches of randomly transformed images.

# It will loop indefinitely, so we need to `break` the loop at some point!

i = 0

for batch in datagen.flow(x, batch_size=1):

    plt.figure(i)

    imgplot = plt.imshow(image.array_to_img(batch[0]))

    i += 1

    if i % 4 == 0:

        break



plt.show()
model = models.Sequential()

model.add(layers.Conv2D(128, (3, 3),

                        activation='relu',

                        input_shape=(150, 150, 3)))

model.add(layers.MaxPooling2D((2, 2)))



model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.add(layers.MaxPooling2D((2, 2)))



model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.add(layers.MaxPooling2D((2, 2)))



model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.add(layers.MaxPooling2D((2, 2)))



model.add(layers.Flatten())

model.add(layers.Dropout(0.2))

model.add(layers.Dense(1500, activation='relu'))

model.add(layers.Dense(700, activation='relu'))

model.add(layers.Dense(350, activation='relu'))

model.add(layers.Dense(1, activation='sigmoid'))



model.compile(loss='binary_crossentropy',

              optimizer=optimizers.RMSprop(lr=0.00005),

              metrics=['acc'])
model.summary()
train_datagen = ImageDataGenerator(

    rescale=1./255,

    rotation_range=40,

    width_shift_range=0.2,

    height_shift_range=0.2,

    shear_range=0.2,

    zoom_range=0.4,

    horizontal_flip=True)



# Note that the validation data should not be augmented!

test_datagen = ImageDataGenerator(rescale=1./255)



train_generator = train_datagen.flow_from_directory(

        # This is the target directory

        train_dir,

        # All images will be resized to 150x150

        target_size=(150, 150),

        batch_size=160,

        # Since we use binary_crossentropy loss, we need binary labels

        class_mode='binary')



validation_generator = test_datagen.flow_from_directory(

        validation_dir,

        target_size=(150, 150),

        batch_size=80,

        class_mode='binary')
history = model.fit_generator(

      train_generator,

      steps_per_epoch=100,

      epochs=100,

      validation_data=validation_generator,

      validation_steps=50)
model.save('CvD_Model_02.h5')
acc = history.history['acc']

val_acc = history.history['val_acc']

loss = history.history['loss']

val_loss = history.history['val_loss']



epochs = range(len(acc))



plt.figure()

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
test_generator = test_datagen.flow_from_directory(

        test_dir,

        target_size=(150, 150),

        batch_size=40,

        class_mode='binary',

        shuffle=False)



test_loss, test_acc = model.evaluate_generator(test_generator, steps=50)

print('Test Accuracy:', test_acc)
from sklearn.metrics import accuracy_score, confusion_matrix

import numpy as np

# load model

# from tensorflow.keras.models import load_model

# model = load_model('/kaggle/working/cats_and_dogs_small_1.h5')

# preds = model.predict_generator(test_generator, steps=len(test_generator))

# preds = model.predict(test_generator)

preds = model.predict_generator(test_generator,steps = len(test_generator.labels//50))



y=test_generator.classes # shape=(2500,)

y_test =y.reshape(2000,1)



acc = accuracy_score(test_generator.labels, np.round(preds))*100

cm = confusion_matrix(test_generator.labels, np.round(preds))



tn, fp, fn, tp = cm.ravel()



print('============TEST METRICS=============')

precision = tp/(tp+fp)*100

recall = tp/(tp+fn)*100

print('Accuracy: {}%'.format(acc))

print('Precision: {}%'.format(precision))

print('Recall: {}%'.format(recall))

print('F1-score: {}'.format(2*precision*recall/(precision+recall)))
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

from sklearn.metrics import confusion_matrix



def plot_cm(y_true, y_pred):

    cm = confusion_matrix(y_true, y_pred, labels=np.unique(y_true))

    cm_sum = np.sum(cm, axis=1, keepdims=True)

    cm_perc = cm / cm_sum.astype(float) * 100

    annot = np.empty_like(cm).astype(str)

    nrows, ncols = cm.shape

    for i in range(nrows):

        for j in range(ncols):

            c = cm[i, j]

            p = cm_perc[i, j]

            if i == j:

                s = cm_sum[i]

                annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)

            elif c == 0:

                annot[i, j] = ''

            else:

                annot[i, j] = '%.1f%%\n%d' % (p, c)

    cm = pd.DataFrame(cm, index=np.unique(y_true), columns=np.unique(y_true))

    cm.index.name = 'Actual'

    cm.columns.name = 'Predicted'

    fig, ax = plt.subplots()

    sns.heatmap(cm, cmap= "YlGnBu", annot=annot, fmt='', ax=ax)

    

plot_cm(test_generator.labels, np.round(preds))