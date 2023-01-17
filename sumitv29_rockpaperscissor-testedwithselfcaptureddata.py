# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import shutil

import os

import tensorflow as tf





# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

# BASE_DATASET_DIR = "../input/rockpaperscissors/"

if os.path.exists('/kaggle/working/rockpaperscissors'):

    shutil.rmtree('/kaggle/working/rockpaperscissors')

shutil.copytree("../input/rockpaperscissors/", '/kaggle/working/rockpaperscissors')



BASE_DATASET_DIR = '/kaggle/working/rockpaperscissors'

# Folder are created because such type of structure are useful when using KERAS ImageDataGenerator, which takes care of

# data augmentation like rotation, scaling, shearing, flipping etc. which we don't have to do seperately. 

# We will keep this option of data augmentation in case our model overfits. Using ImageDataGenerator we can create

# more data on the fly while training and saving disk space. We will experience this when we use ImageDataGenerator.

import os

if not os.path.exists('/kaggle/working/validation'):

    os.mkdir('/kaggle/working/validation')

    os.mkdir('/kaggle/working/validation/rock')

    os.mkdir('/kaggle/working/validation/paper')

    os.mkdir('/kaggle/working/validation/scissors')

    

# under testing dataset folder we don't need any folder structure. because out trained model will predict the

# images store belongs to which class

if not os.path.exists('/kaggle/working/testing'):

    os.mkdir('/kaggle/working/testing')



VALIDATION_DATASET_DIR = '/kaggle/working/validation/'

TESTING_DATASET_DIR = '/kaggle/working/testing/'
print(tf.__version__)


shutil.rmtree(BASE_DATASET_DIR+'/rps-cv-images')

print(BASE_DATASET_DIR)

print(os.listdir(VALIDATION_DATASET_DIR))

print(os.listdir(BASE_DATASET_DIR))

# print(os.listdir(TESTING_DATASET_DIR))

# Get total count of dataset

print(len(os.listdir("/kaggle/working/rockpaperscissors/rock")))

print(len(os.listdir("/kaggle/working/rockpaperscissors/scissors")))

print(len(os.listdir("/kaggle/working/rockpaperscissors/paper")))
# Extracting the file names in a list, shuffling and dividing into test and validation set.

# We will keep the split as 20% for validation and 10% for testing.

import random

from tqdm.notebook import trange

from shutil import move



def clear_dir(folder):

    """ Deletes the content of the folder to avoid adding files when the code is run multiple times"""

    if len(os.listdir(folder)) != 0:

        shutil.rmtree(folder)

        os.mkdir(folder)



def split_data(src_folder_path, dst_folder_path, split_range):

    """

    src_folder_path: Folder path we want to split

    dst_folder_path: Destination Path

    split_range: percentage of data to split

    """

    if src_folder_path is not None and dst_folder_path is not None:

        if os.path.exists(src_folder_path) and os.path.exists(dst_folder_path):

            print("Splitting data for {}".format(dst_folder_path))

            

            file_names = [f_name for f_name in os.listdir(src_folder_path)]

            shuffled_files = random.sample(file_names, len(file_names))

            

            if dst_folder_path != TESTING_DATASET_DIR:

                clear_dir(dst_folder_path)

            

            print("Total Files Found: {}".format(len(file_names)))

            print("Files to split {}".format(round(len(shuffled_files)*split_range)))

            for each_img in trange(round(len(shuffled_files)*split_range)):

                src_full_file_path = os.path.join(src_folder_path, shuffled_files[each_img])

                dst_full_path = os.path.join(dst_folder_path, shuffled_files[each_img])      

                

                move(src_full_file_path, dst_full_path)

                

        else:

            print("Source {} or destination {} folder doesnot exists.".format(src_folder_path, dst_folder_path))

    else:

        print("Source or destination folder arguments missing.")



classes = ['rock', 'paper', 'scissors']

BASE_DATASET_DIR = '/kaggle/working/rockpaperscissors'

for each_labels in classes:    

    label_src = os.path.join(BASE_DATASET_DIR, each_labels)

    label_dst_val = os.path.join(VALIDATION_DATASET_DIR, each_labels)

    label_dst_test = TESTING_DATASET_DIR

    split_data(label_src, label_dst_val, 0.2)

    split_data(label_src, label_dst_test, 0.1)

    



train_rock = os.path.join(BASE_DATASET_DIR, classes[0])

train_paper = os.path.join(BASE_DATASET_DIR, classes[1])

train_scissor = os.path.join(BASE_DATASET_DIR, classes[2])



valid_rock = os.path.join(VALIDATION_DATASET_DIR, classes[0])

valid_paper = os.path.join(VALIDATION_DATASET_DIR, classes[1])

valid_scissor = os.path.join(VALIDATION_DATASET_DIR, classes[2])

# VERIFY WHETHER THE FOLDERS ARE CREATED OR NOT

print("Training Dataset")



print(len(os.listdir(train_rock)))

print(len(os.listdir(train_paper)))

print(len(os.listdir(train_scissor)))



print("Validation dataset count")



print(len(os.listdir(valid_rock)))

print(len(os.listdir(valid_paper)))

print(len(os.listdir(valid_scissor)))



print("Testing dataset count")

print(len(os.listdir(TESTING_DATASET_DIR)))
%matplotlib inline

import matplotlib.pyplot as plt

import matplotlib.image as mpimg



fig = plt.gcf()

n_rows, n_cols = 3, 5



# setting the canvas size to fit all the images

fig.set_size_inches(n_cols*5, n_rows*5)

rock_fnames = [os.path.join(train_rock,images) for images in os.listdir(train_rock)][0:5]

paper_fnames = [os.path.join(train_paper,images) for images in os.listdir(train_paper)][0:5]

scissor_fnames = [os.path.join(train_scissor,images) for images in os.listdir(train_scissor)][0:5]



for idx, img_path  in enumerate(rock_fnames + paper_fnames + scissor_fnames):

    # Set up subplot; subplot indices start at 1

    sp = plt.subplot(n_rows, n_cols, idx + 1)

    fig.suptitle("RPS: Training Images", fontsize=48)

    sp.axis('Off') # Don't show axes (or gridlines)

    img = mpimg.imread(img_path)

    plt.imshow(img)

plt.show()

fig = plt.gcf()

n_rows, n_cols = 3, 5



# setting the canvas size to fit all the images

fig.set_size_inches(n_cols*5, n_rows*5)

rock_fnames = [os.path.join(valid_rock,images) for images in os.listdir(valid_rock)][0:5]

paper_fnames = [os.path.join(valid_paper,images) for images in os.listdir(valid_paper)][0:5]

scissor_fnames = [os.path.join(valid_scissor,images) for images in os.listdir(valid_scissor)][0:5]



for idx, img_path  in enumerate(rock_fnames + paper_fnames + scissor_fnames):

    # Set up subplot; subplot indices start at 1

    sp = plt.subplot(n_rows, n_cols, idx + 1)

    fig.suptitle("RPS: Validation Images", fontsize=48)

    sp.axis('Off') # Don't show axes (or gridlines)

    img = mpimg.imread(img_path)

    plt.imshow(img)

plt.show()
fig = plt.gcf()

n_rows, n_cols = 5, 5

fig.set_size_inches(n_cols*5, n_rows*5)



tst_fnames = [os.path.join(TESTING_DATASET_DIR,images) for images in os.listdir(TESTING_DATASET_DIR)][0:25]

tst_fnames = random.sample(tst_fnames, len(tst_fnames))



for idx, img_path  in enumerate(tst_fnames):

    # Set up subplot; subplot indices start at 1

    sp = plt.subplot(n_rows, n_cols, idx + 1)

    fig.suptitle("RPS: TESTING Images", fontsize=40)

    sp.axis('Off') # Don't show axes (or gridlines)

    img = mpimg.imread(img_path)

    plt.imshow(img)

plt.show()
from tensorflow.keras.preprocessing.image import ImageDataGenerator



# Just normalizing the data now.

train_datagen = ImageDataGenerator(rescale=1/255,

                                   width_shift_range = 0.1,

                                   height_shift_range = 0.1,

                                   rotation_range=0.1,

                                   zoom_range=0.1,

                                   fill_mode="nearest",

                                   vertical_flip=True,

                                   horizontal_flip=True

                                  )



train_generator = train_datagen.flow_from_directory('/kaggle/working/rockpaperscissors',

                                                    target_size=(150, 150),

                                                    class_mode="categorical",

                                                    batch_size=32)

valid_datagen = ImageDataGenerator(rescale=1/255)

valid_generator = valid_datagen.flow_from_directory(VALIDATION_DATASET_DIR,

                                                    class_mode="categorical",

                                                    target_size=(150,150),

                                                    batch_size=32)
import tensorflow as tf

rps_model = tf.keras.models.Sequential([

    tf.keras.layers.Conv2D(64, (3,3), activation="relu", input_shape=(150,150,3)),    

    tf.keras.layers.MaxPool2D(2,2),

    tf.keras.layers.Dropout(0.5),

    tf.keras.layers.Conv2D(64, (3,3), activation="relu"),

    tf.keras.layers.MaxPool2D(2,2),

    tf.keras.layers.Conv2D(32, (3,3), activation="relu"),

    tf.keras.layers.MaxPool2D(2,2),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(512, activation='relu'),

    tf.keras.layers.Dropout(0.5),

    tf.keras.layers.Dense(3, activation='softmax')   

])

rps_model.summary()
class myCallBack(tf.keras.callbacks.Callback):

    def on_epoch_end(self, epoch, logs={}):

        print(logs)

        if(logs.get('accuracy') > 0.97):

            print("\nReached 97% accuracy so cancelling training!")

            self.model.stop_training = True

from tensorflow.keras.optimizers import RMSprop



rps_model.compile(optimizer=RMSprop(lr=0.0001),

                  loss='categorical_crossentropy',

                  metrics=['accuracy'])
epochs = 30

training_len = 1576

batch_size = 32

steps_per_epoch=int(training_len/batch_size)



callbacks = myCallBack()

history = rps_model.fit_generator(train_generator,

                                  steps_per_epoch=steps_per_epoch,

                                  epochs=epochs,

                                  validation_data=valid_generator,

                                  callbacks=[callbacks],

                                  verbose=1)
acc = history.history['accuracy']

val_acc = history.history['val_accuracy']

loss = history.history['loss']

val_loss = history.history['val_loss']



epochs = range(len(acc))



plt.plot(epochs, acc, 'r', label='Training accuracy')

plt.plot(epochs, val_acc, 'b', label='Validation accuracy')

plt.title('Training and validation accuracy')

plt.legend(loc=0)

plt.figure()





plt.show()
classes = ['paper', 'rock', 'scissors']

label_map = train_generator.class_indices

print(label_map)



def test_model(image):

    import cv2

    img = image

    resized = cv2.resize(img, (150,150), interpolation = cv2.INTER_AREA)

    exp_img = np.expand_dims(resized, axis=0)

    y_prob = rps_model.predict(exp_img)

    _cls = y_prob.argmax(axis=-1)

#     label_map = train_generator.class_indices

#     print(y_prob, _cls)

#     print(label_map)

#     sp = plt.subplot(1, 1, 1, title="Label: "+ classes[_cls[0]])

#     fig.suptitle("RPS: Self-Captured data to test", fontsize=40)

#     sp.axis('Off') # Don't show axes (or gridlines)     

#     plt.imshow(img)

    return classes[_cls[0]]

new_dataset = '../input/rpsselfcaptureddata/'

stst_fnames = [os.path.join(new_dataset,images) for images in os.listdir(new_dataset)]

stst_fnames = random.sample(stst_fnames, len(stst_fnames))
fig = plt.gcf()

n_rows, n_cols = 5,5

fig.set_size_inches(n_cols*5, n_rows*5)

for idx, img in enumerate(tst_fnames):

    img = mpimg.imread(img)

    label_ = str(test_model(img))

    sp = plt.subplot(n_cols, n_rows, idx + 1, title="Label: "+ label_)

    fig.suptitle("Inference", fontsize=35)

    sp.axis('Off') # Don't show axes (or gridlines)  

    plt.imshow(img)

plt.show()



fig = plt.gcf()

n_rows, n_cols = 2,5

fig.set_size_inches(n_cols*3, n_rows*3)

for idx, img in enumerate(stst_fnames):

    img = mpimg.imread(img)

    label_ = str(test_model(img))

    sp = plt.subplot(n_rows, n_cols, idx + 1, title="Label: "+ label_)

    fig.suptitle("Inference", fontsize=35)

    sp.axis('Off') # Don't show axes (or gridlines)  

    plt.imshow(img)

plt.show()
import numpy as np

import random

from   tensorflow.keras.preprocessing.image import img_to_array, load_img



# Let's define a new Model that will take an image as input, and will output

# intermediate representations for all layers in the previous model after

# the first.

successive_outputs = [layer.output for layer in rps_model.layers[1:]]



#visualization_model = Model(img_input, successive_outputs)

visualization_model = tf.keras.models.Model(inputs = rps_model.input, outputs = successive_outputs)



tr_imgs = [rock_fnames[:2], paper_fnames[:2], scissor_fnames[0:2]]



img_path = random.choice(tr_imgs)

img = load_img(img_path[0], target_size=(150, 150))  # this is a PIL image



x   = img_to_array(img)                           # Numpy array with shape (150, 150, 3)

x   = x.reshape((1,) + x.shape)                   # Numpy array with shape (1, 150, 150, 3)



# Rescale by 1/255

x /= 255.0



# Let's run our image through our network, thus obtaining all

# intermediate representations for this image.

successive_feature_maps = visualization_model.predict(x)



# These are the names of the layers, so can have them as part of our plot

layer_names = [layer.name for layer in rps_model.layers]



# -----------------------------------------------------------------------

# Now let's display our representations

# -----------------------------------------------------------------------

for layer_name, feature_map in zip(layer_names, successive_feature_maps):

  

  if len(feature_map.shape) == 4:

    

    #-------------------------------------------

    # Just do this for the conv / maxpool layers, not the fully-connected layers

    #-------------------------------------------

    n_features = feature_map.shape[-1]  # number of features in the feature map

    size       = feature_map.shape[ 1]  # feature map shape (1, size, size, n_features)

    

    # We will tile our images in this matrix

    display_grid = np.zeros((size, size * n_features))

    

    #-------------------------------------------------

    # Postprocess the feature to be visually palatable

    #-------------------------------------------------

    for i in range(n_features):

      x  = feature_map[0, :, :, i]

      x -= x.mean()

      x /= x.std ()

      x *=  64

      x += 128

      x  = np.clip(x, 0, 255).astype('uint8')

      display_grid[:, i * size : (i + 1) * size] = x # Tile each filter into a horizontal grid



    #-----------------

    # Display the grid

    #-----------------



    scale = 20. / n_features

    plt.figure( figsize=(scale * n_features, scale) )

    plt.title ( layer_name )

    plt.grid  ( False )

    plt.imshow( display_grid, aspect='auto', cmap='viridis' ) 