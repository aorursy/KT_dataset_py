import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import matplotlib.pyplot as plt

from PIL import Image

import random

import tensorflow as tf

from tensorflow.keras.optimizers import RMSprop

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from shutil import copyfile, rmtree
INPUT_ROOT = "../input/"

def from_input(path):

    return os.path.join(INPUT_ROOT, path)
train_info = pd.read_csv(from_input("Train.csv"))

train_info.head()
train_info.describe()
test_info = pd.read_csv(from_input("Test.csv"))

test_info.head()
test_info.describe()
VALIDATION_RATIO = 0.2

TRAINING_DIR = '/tmp/ts_train/'

VALIDATION_DIR = '/tmp/ts_val'

TEST_DIR = '/tmp/ts_test'
def copy_files(srcdir, dstdir, file_names):

    for file in file_names:

        src_file_path = os.path.join(srcdir, file)

        if os.path.getsize(src_file_path) > 0:

            try:

                # Check if image if currupt by by trying to open and flip it.

                im = Image.open(src_file_path)

                im.transpose(Image.FLIP_LEFT_RIGHT)

                im.close()

                copyfile(src_file_path, os.path.join(dstdir, file))

            except Exception as e:

                print(e)

                print("{} is corrupt, skipping".format(file))

        else:    

            print("{} is corrupt, skipping".format(file))
split_size = 1 - VALIDATION_RATIO



for i in range(43):

    print('\rCopying images of class {}'.format(i), end = '\r')

    dst_train_class_dir = os.path.join(TRAINING_DIR, str(i))

    if os.path.isdir(dst_train_class_dir):

        rmtree(dst_train_class_dir)

    os.makedirs(dst_train_class_dir)    

    dst_val_class_dir = os.path.join(VALIDATION_DIR, str(i))

    if os.path.isdir(dst_val_class_dir):

        rmtree(dst_val_class_dir)

    os.makedirs(dst_val_class_dir)

    src_class_dir = os.path.join(from_input('train'), str(i))

    file_names = os.listdir(src_class_dir)

    # Use custom generator to avoid real randomness

    # on multiple executions.

    randgen = random.Random(29)

    randgen.shuffle(file_names)

    train_size = int(len(file_names) * split_size)

    train_file_names = file_names[:train_size]

    val_file_names = file_names[train_size:]

    copy_files(src_class_dir, dst_train_class_dir, train_file_names)

    copy_files(src_class_dir, dst_val_class_dir, val_file_names)

    

print('Done copying files                            ')
for i in range(43):

    dir = os.path.join(VALIDATION_DIR, str(i))

    print('{} contains {} images'.format(dir, len(os.listdir(dir))))

    dir = os.path.join(TRAINING_DIR, str(i))

    print('{} contains {} images'.format(dir, len(os.listdir(dir))))

%matplotlib inline



import matplotlib.image as mpimg

import matplotlib.pyplot as plt



# Parameters for our graph; we'll output images in a 4x4 configuration

nrows = 8

ncols = 6



pic_offset = 0 # Index for iterating over images

vpic_offset = 0 # Index for iterating over validation images
def show_images(dir, offset):

    # Set up matplotlib fig, and size it to fit 4x4 pics

    fig = plt.gcf()

    fig.set_size_inches(ncols*3, nrows*3)



    for i in range(43):

        # Set up subplot; subplot indices start at 1

        sp = plt.subplot(nrows, ncols, i + 1)

        sp.axis('Off') # Don't show axes (or gridlines)

        subdir = os.path.join(dir, str(i))

        files = os.listdir(subdir)

        img_path = os.path.join(subdir, files[offset % len(files)])

        img = mpimg.imread(img_path)

        #print(img.shape)

        plt.imshow(img)



    plt.show()
show_images(TRAINING_DIR, pic_offset)

pic_offset += 1

show_images(VALIDATION_DIR, vpic_offset)

vpic_offset += 1
class StopOnAccReachedCallback(tf.keras.callbacks.Callback):

    def on_epoch_end(self, epoch, logs={}):

        if logs['acc'] >= 0.999 and logs['val_acc'] >= 0.999:

            self.model.stop_training = True

            print("\nReached accuracy {} at epoch {}, stopping...".format(logs['acc'], epoch))
TARGET_SIZE = (40, 40)

model = tf.keras.models.Sequential([

    tf.keras.layers.Flatten(input_shape=TARGET_SIZE + (3,)),

    tf.keras.layers.Dense(256, activation='relu'),

    tf.keras.layers.Dense(128, activation='relu'),

    tf.keras.layers.Dense(43, activation='softmax')

])



model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])



model.summary()
BATCH_SIZE = 300

classes = [str(i) for i in range(43)]



train_datagen = ImageDataGenerator(rescale=1.0/255.0)

train_generator = train_datagen.flow_from_directory(TRAINING_DIR,

                                                   target_size=TARGET_SIZE,

                                                    batch_size=BATCH_SIZE,

                                                    shuffle=True,

                                                    seed=17,

                                                    classes=classes,

                                                    class_mode='categorical')



validation_datagen = ImageDataGenerator(rescale=1.0/255.0)

validation_generator = validation_datagen.flow_from_directory(VALIDATION_DIR,

                                                   target_size=TARGET_SIZE,

                                                    batch_size=BATCH_SIZE, #2502,

                                                    shuffle=False,

                                                    classes=classes,

                                                    class_mode='categorical')
history = model.fit_generator(train_generator,

                              epochs=20,

                              verbose=1,

                              callbacks=[

                                  tf.keras.callbacks.EarlyStopping(monitor='acc', min_delta=0.0001, patience=2),

                                  StopOnAccReachedCallback()

                              ],

                              validation_data=validation_generator)
# PLOT LOSS AND ACCURACY

%matplotlib inline



import matplotlib.image  as mpimg

import matplotlib.pyplot as plt



#-----------------------------------------------------------

# Retrieve a list of list results on training and test data

# sets for each training epoch

#-----------------------------------------------------------

acc=history.history['acc']

val_acc=history.history['val_acc']

loss=history.history['loss']

val_loss=history.history['val_loss']



epochs=range(len(acc)) # Get number of epochs



#------------------------------------------------

# Plot training and validation accuracy per epoch

#------------------------------------------------

plt.plot(epochs, acc, 'r', "Training Accuracy")

plt.plot(epochs, val_acc, 'b', "Validation Accuracy")

plt.title('Training and validation accuracy')

plt.xlabel('Epoch')

plt.figure()



#------------------------------------------------

# Plot training and validation loss per epoch

#------------------------------------------------

plt.plot(epochs, loss, 'r', "Training Loss")

plt.plot(epochs, val_loss, 'b', "Validation Loss")

plt.xlabel('Epoch')

plt.title('Training and validation loss')
#Predicting with the test data

paths = test_info['Path'].values

y_test = test_info['ClassId'].values



data=[]

    

src_class_dir = from_input("test")

#file_names = os.listdir(src_class_dir)

for i in range(43):

    print('\rCopying images of class {}'.format(i), end = '\r')

    dst_test_class_dir = os.path.join(TEST_DIR, str(i))

    if os.path.isdir(dst_test_class_dir):

        rmtree(dst_test_class_dir)

    os.makedirs(dst_test_class_dir)    

    # Use custom generator to avoid real randomness

    # on multiple executions.

    file_names = [f.replace('Test/', '') for (j, f) in enumerate(paths) if y_test[j] == i]

    copy_files(src_class_dir, dst_test_class_dir, file_names)



test_datagen = ImageDataGenerator(rescale=1.0/255.0)

test_generator = test_datagen.flow_from_directory(TEST_DIR,

                                                   target_size=TARGET_SIZE,

                                                    batch_size=BATCH_SIZE,

                                                    classes=classes,

                                                    class_mode='categorical')



print(model.metrics_names)

print(model.evaluate_generator(test_generator))
#Predicting with the test data

paths = test_info['Path'].values

y_test = test_info['ClassId'].values

from tensorflow.keras.utils import to_categorical

y_test = to_categorical(y_test, 43)



data=[]

#resized_image = None

for f in paths:

    image = Image.open(os.path.join(from_input('test'), f.replace('Test/', '')))

    resized_image = image.resize(TARGET_SIZE)

    data.append(np.array(resized_image))



X_test = np.array(data).astype('float32') / 255.0 



result = model.evaluate(X_test, y_test)

print(model.metrics_names)

print(result)