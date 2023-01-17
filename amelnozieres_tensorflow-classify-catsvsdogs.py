import numpy as np 

import pandas as pd 

import tensorflow as tf

import os

import zipfile

import random

import shutil

from tensorflow.keras.optimizers import RMSprop

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from shutil import copyfile

from os import getcwd







import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))





print(os.listdir("../input"))



path_cats_and_dogs = "../input/dogs-vs-cats/train.zip"

#shutil.rmtree('/tmp')



local_zip = path_cats_and_dogs

zip_ref = zipfile.ZipFile(local_zip, 'r')

zip_ref.extractall('.')

zip_ref.close()

path_cats_and_dogs = "../input/dogs-vs-cats/test1.zip"

#shutil.rmtree('/tmp')



local_zip = path_cats_and_dogs

zip_ref = zipfile.ZipFile(local_zip, 'r')

zip_ref.extractall('.')

zip_ref.close()
try:

    main_dir = "/kaggle/working/"

    train_dir = "train"

    val_dir = "val"



    train_dir = os.path.join(main_dir,train_dir)

    # Directory with our training cat/dog pictures

    train_cats_dir = os.path.join(train_dir, 'cats')

    train_dogs_dir = os.path.join(train_dir, 'dogs')

    os.mkdir(train_cats_dir)

    os.mkdir(train_dogs_dir)

    # Directory with our validation cat/dog pictures

    val_dir = os.path.join(main_dir,"val")

    os.mkdir(val_dir)

    val_cats_dir = os.path.join(val_dir, 'cats')

    val_dogs_dir = os.path.join(val_dir, 'dogs')

    os.mkdir(val_cats_dir)

    os.mkdir(val_dogs_dir)



except OSError:

    pass
##let's put the cats images in the cats directory and the dogs in the dogs directory

# for the train directory we parse the jpg name if the name start with cat we put it in the cats dir

main_dir = "/kaggle/working/"

train_dir = "train"

train_path = os.path.join(main_dir,train_dir)



prefixed_dogs = [filename for filename in os.listdir(train_path) if filename.startswith("dog.")]

print(len(prefixed_dogs))

prefixed_cats = [filename for filename in os.listdir(train_path) if filename.startswith("cat.")]

print(len(prefixed_cats))



def move_files(src_file):

    

    for filename in prefixed_dogs:

        shutil.move(src_file+filename, src_file+'dogs/'+filename)

        

    for filename in prefixed_cats:

        shutil.move(src_file+filename, src_file+'cats/'+filename)

    



move_files("/kaggle/working/train/")







print(len(os.listdir('/kaggle/working/train/dogs')))

print(len(os.listdir('/kaggle/working/train/cats')))

print(len(os.listdir('/kaggle/working/train')))

def split_data(SOURCE, VALID, SPLIT_SIZE):

# This funtion takes as argument:

###SOURCE : the directory's path of images that will be splitted

###VALID : the directory's path of the validation receiving the dogs or the cats images

###SPLIT_SIZE: the size of the split. 0.9 means 90% of cats images will remain in train/cats and 10% will be moved to the validation directory's cats 

###and the same will be done to the dogs images

    SRC_files = [f for f in os.listdir(SOURCE) if os.path.isfile(os.path.join(SOURCE, f))]

    SRC_Size = len(SRC_files)

    #print(SRC_Size)

    if SRC_Size != 0:

        # we shuffle the images before the split

        shuffled_files = random.sample(SRC_files, len(SRC_files))

        #print("shuffled")

        TRN_size = int(SRC_Size * SPLIT_SIZE)

        VAL_SIZE = int(SRC_Size - TRN_size)

        print(TRN_size)

        train_set = shuffled_files[0:TRN_size]

        val_set = shuffled_files[-VAL_SIZE:SRC_Size]

        for filename in val_set:

            if os.path.getsize(SOURCE+filename)!=0:

                shutil.move(SOURCE+filename, VALID+filename)

            else:

                print(filename + ' is zero length. So ignoring!')

                pass





                    

CAT_SOURCE_DIR = "/kaggle/working/train/cats/"

TESTING_CATS_DIR = "/kaggle/working/val/cats/"



DOG_SOURCE_DIR = "/kaggle/working/train/dogs/"

TESTING_DOGS_DIR = "/kaggle/working/val/dogs/"



split_size = .9

split_data(CAT_SOURCE_DIR, TESTING_CATS_DIR, split_size)

split_data(DOG_SOURCE_DIR, TESTING_DOGS_DIR, split_size)
print(len(os.listdir('/kaggle/working/train/dogs')))

print(len(os.listdir('/kaggle/working/train/cats')))

print(len(os.listdir('/kaggle/working/train')))

print(len(os.listdir('/kaggle/working/val/dogs')))

print(len(os.listdir('/kaggle/working/val/cats')))
# DEFINE A KERAS MODEL TO CLASSIFY CATS V DOGS

# USE AT LEAST 3 CONVOLUTION LAYERS

IMAGE_WIDTH=150

IMAGE_HEIGHT=150

IMAGE_SIZE=(IMAGE_WIDTH, IMAGE_HEIGHT)



model = tf.keras.models.Sequential([

# YOUR CODE HERE

    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, 3)),

    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),

    tf.keras.layers.MaxPooling2D(2,2), 

    tf.keras.layers.Conv2D(64, (3,3), activation='relu'), 

    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(512, activation='relu'), 

    tf.keras.layers.Dense(1, activation='sigmoid')  

])



model.compile(optimizer=RMSprop(lr=0.001), loss='binary_crossentropy', metrics=['acc'])
TRAINING_DIR = '/kaggle/working/train'

train_datagen = ImageDataGenerator(rescale=1./255)

#       rotation_range=40,

#       width_shift_range=0.2,

#       height_shift_range=0.2,

#       shear_range=0.2,

#       zoom_range=0.2,

#       horizontal_flip=True,

#       fill_mode='nearest')#YOUR CODE HERE



# NOTE: YOU MUST USE A BATCH SIZE OF 10 (batch_size=10) FOR THE 

# TRAIN GENERATOR.

train_generator = train_datagen.flow_from_directory(TRAINING_DIR,

                                                    batch_size=10,

                                                    class_mode='binary',

                                                    target_size=(150, 150))



VALIDATION_DIR = '/kaggle/working/val'#YOUR CODE HERE

validation_datagen = ImageDataGenerator( rescale = 1.0/255. )#YOUR CODE HERE



# NOTE: YOU MUST USE A BACTH SIZE OF 10 (batch_size=10) FOR THE 

# VALIDATION GENERATOR.

validation_generator = validation_datagen.flow_from_directory(VALIDATION_DIR,

                                                         batch_size=10,

                                                         class_mode  = 'binary',

                                                         target_size = (150, 150))







history = model.fit_generator(train_generator,

                              epochs=2,

                              verbose=1,

                              validation_data=validation_generator)
model.save_weights("model.h5")
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

plt.figure()



#------------------------------------------------

# Plot training and validation loss per epoch

#------------------------------------------------

plt.plot(epochs, loss, 'r', "Training Loss")

plt.plot(epochs, val_loss, 'b', "Validation Loss")





plt.title('Training and validation loss')
test = os.listdir('/kaggle/working/test1')

print(type(test))



# preprocessing test

TEST_DIR  = '/kaggle/working/test1'

test_df = pd.DataFrame({'filename': test})



nb_samples = test_df.shape[0]



test_gen = ImageDataGenerator(rescale=1./255)

test_generator = test_gen.flow_from_dataframe(

    test_df, 

    TEST_DIR, 

    x_col='filename',

    y_col=None,

    class_mode=None,

    target_size=(150,150),

    batch_size = 10

)







test_df
predict = model.predict_generator(test_generator)
predict[2]
test_df['prediction'] = predict


test_df
from keras.preprocessing.image import ImageDataGenerator, load_img

import matplotlib.pyplot as plt

img = load_img('/kaggle/working/test1/10392.jpg', target_size=(150,150))

plt.imshow(img)
submission_df = test_df.copy()
submission_df['id'] = submission_df['filename'].str.split('.').str[0]

submission_df["label"] =  np.where(submission_df['prediction'] >0.7, 1, 0)

submission_df.drop(['filename','prediction'], axis=1, inplace=True)

submission_df.to_csv('submission.csv', index=False)

submission_df

submission_df.to_csv('submission.csv', index=False)