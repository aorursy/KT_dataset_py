# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import matplotlib.image as mpimg

from PIL import Image

 

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import random, os

# Any results you write to the current directory are saved as output.
sample_submission_df = pd.read_csv("/kaggle/input/cgiar-computer-vision-for-crop-disease/sample_submission.csv")

sample_submission_df.shape
train_dir = '/kaggle/input/cgiar-computer-vision-for-crop-disease/train/train'

test_dir = '/kaggle/input/cgiar-computer-vision-for-crop-disease/test/test'

train_leaf_rust_dir = '/kaggle/input/cgiar-computer-vision-for-crop-disease/train/train/leaf_rust'

train_stem_rust_dir = '/kaggle/input/cgiar-computer-vision-for-crop-disease/train/train/stem_rust'

train_healthy_wheat_dir = '/kaggle/input/cgiar-computer-vision-for-crop-disease/train/train/healthy_wheat'
# directory to store processed images

train_new_dir = '/kaggle/working/train'

test_new_dir = '/kaggle/working/test'

train_new_leaf_rust_dir = '/kaggle/working/train/leaf_rust'

train_new_stem_rust_dir = '/kaggle/working/train/stem_rust'

train_new_healthy_wheat_dir = '/kaggle/working/train/healthy_wheat'

os.mkdir(train_new_dir)

os.mkdir(test_new_dir)

os.mkdir(train_new_leaf_rust_dir)

os.mkdir(train_new_stem_rust_dir)

os.mkdir(train_new_healthy_wheat_dir)
# 7U06EV.gif

def process_img(old_dir, new_dir):

    for dirname, _, filenames in os.walk(old_dir):

        dir_p = new_dir + '/'

        for filename in filenames:

            filename_name = filename.split('.')[0]

            filename_ext = filename.split('.')[-1].lower()

            if filename_ext != "gif":

                img = Image.open(dirname + '/' + filename)

            else:

                img = Image.open(dirname + '/' + filename).convert('RGB')

            path = dir_p + filename_name + '.jpeg'

            img.save(path)
process_img(test_dir, test_new_dir)

process_img(train_leaf_rust_dir, train_new_leaf_rust_dir)

process_img(train_stem_rust_dir, train_new_stem_rust_dir)

process_img(train_healthy_wheat_dir, train_new_healthy_wheat_dir)
test_image_paths = []

for dirname, _, filenames in os.walk(test_new_dir):

    for filename in filenames:

        path = dirname + "/" + filename

        test_image_paths.append(path)

test_image_paths
# labeling trainging data in a dataframe

train_df = pd.DataFrame(columns = ["ID", "leaf_rust", "stem_rust", "healthy_wheat", "path"])

for dirname, _, filenames in os.walk(train_new_dir):

    for filename in filenames:

        path = dirname + "/" + filename

        category = dirname.split("/")[-1]

        ID = filename.split(".")[0]

        if category == "healthy_wheat":

            train_df = train_df.append([{ 'ID': ID, 'leaf_rust': 0, 'stem_rust': 0, 'healthy_wheat':1, 'path':path}])

        elif category == "stem_rust":

            train_df = train_df.append([{ 'ID': ID, 'leaf_rust': 0, 'stem_rust': 1, 'healthy_wheat':0, 'path':path}])

        else:

            train_df = train_df.append([{ 'ID': ID, 'leaf_rust': 1, 'stem_rust': 0, 'healthy_wheat':0, 'path':path}])

train_df = train_df.reset_index(drop=True)

train_df.head()
# displaying sample images of different categories

train_stem_rust_dir_img = train_stem_rust_dir + "/" + random.choice([x for x in os.listdir(train_stem_rust_dir)

               if os.path.isfile(os.path.join(train_stem_rust_dir, x))])

train_leaf_rust_dir_img = train_leaf_rust_dir + "/" + random.choice([x for x in os.listdir(train_leaf_rust_dir)

               if os.path.isfile(os.path.join(train_leaf_rust_dir, x))])

train_healthy_wheat_dir_img = train_healthy_wheat_dir + "/" + random.choice([x for x in os.listdir(train_healthy_wheat_dir)

               if os.path.isfile(os.path.join(train_healthy_wheat_dir, x))])
# Healthy Wheat Image

img = mpimg.imread(train_healthy_wheat_dir_img)

plt.imshow(img)
# Leaf Rust Wheat Image

img = mpimg.imread(train_leaf_rust_dir_img)

plt.imshow(img)
# Stem Rust Wheat Image

img = mpimg.imread(train_stem_rust_dir_img)

plt.imshow(img)
num_train_stem_rust_dir_img = len(os.listdir(train_new_stem_rust_dir))

num_train_leaf_rust_dir_img = len(os.listdir(train_new_leaf_rust_dir))

num_train_healthy_wheat_dir_img = len(os.listdir(train_new_healthy_wheat_dir))





total_train = num_train_stem_rust_dir_img + num_train_leaf_rust_dir_img + num_train_healthy_wheat_dir_img



print('total training stem rust images:', num_train_stem_rust_dir_img)

print('total training leaf rust images:', num_train_leaf_rust_dir_img)

print('total training heathy wheat images:', num_train_healthy_wheat_dir_img)

print("--")

print("Total training images:", total_train)
batch_size = 32

epochs = 10

IMG_HEIGHT = 300

IMG_WIDTH = 300
from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D

from tensorflow.keras.preprocessing.image import ImageDataGenerator
data_generator = ImageDataGenerator(rescale=1./255, validation_split=0.33) # Generator for our training data

train_data_gen = data_generator.flow_from_directory(train_new_dir, target_size=(IMG_HEIGHT, IMG_WIDTH), shuffle=True, seed=13,

                                                     class_mode='categorical', batch_size=batch_size, subset="training")



validate_data_gen = data_generator.flow_from_directory(train_new_dir, target_size=(IMG_HEIGHT, IMG_WIDTH), shuffle=True, seed=13,

                                                     class_mode='categorical', batch_size=batch_size, subset="validation")
sample_training_images, _ = next(train_data_gen)
# This function will plot images in the form of a grid with 1 row and 5 columns where images are placed in each column.

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

    Dense(1, activation='softmax')

])
model.compile(optimizer='adam',

              loss='binary_crossentropy',

              metrics=['accuracy'])
model.summary()
history = model.fit_generator(

    train_data_gen,

    steps_per_epoch=total_train // batch_size,

    epochs=epochs,

    validation_data=validate_data_gen,

    validation_steps=(876-total_train)// batch_size,

    use_multiprocessing=True,

)
acc = history.history['accuracy']

val_acc = history.history['val_accuracy']



loss = history.history['loss']

val_loss = history.history['val_loss']



epochs_range = range(epochs)



plt.figure(figsize=(8, 8))

plt.subplot(1, 2, 1)

plt.plot(epochs_range, acc, label='Training Accuracy')

plt.plot(epochs_range, val_acc, label='Validation Accuracy')

plt.legend(loc='lower right')

plt.title('Training and Validation Accuracy')



plt.subplot(1, 2, 2)

plt.plot(epochs_range, loss, label='Training Loss')

plt.plot(epochs_range, val_loss, label='Validation Loss')

plt.legend(loc='upper right')

plt.title('Training and Validation Loss')

plt.show()