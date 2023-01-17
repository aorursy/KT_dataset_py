# Open this notebook on google colab or kaggle kernel

# Change runtime type to GPU

# In December 2019, tensorflow 2.0 is only available in google colab or kaggle kernel

# ! pip install tensorflow==2.0.0
# Import the relevant libraries

import os

import json

import zipfile

import shutil

import random

import numpy as np

import pandas as pd

import tensorflow as tf



from tensorflow.keras import layers

from tensorflow.keras import Model

from keras.preprocessing.image import ImageDataGenerator

from keras.preprocessing.image import load_img

from shutil import copyfile



print(tf.__version__)
# Import the inception V3 model  

from tensorflow.keras.applications.inception_v3 import InceptionV3



# Clear any exiting model

tf.keras.backend.clear_session()



pre_trained_model = InceptionV3(input_shape = (150,150,3), # reshape images to 150 by 150 by 3 channels

                                include_top = False, # get straight to the CNN layer

                                weights = 'imagenet') # use the builed-in weight pre-trained on imagent



# Make all the layers in the pre-trained model non-trainable/frozen

for layer in pre_trained_model.layers:

  layer.trainable = False
# Print the InceptionV3 model summary

# pre_trained_model.summary()
# Pick one layer from the miffde of the model

# We well build new dense layers after this layer



last_layer = pre_trained_model.get_layer('mixed7')

print('last layer output shape: ', last_layer.output_shape)

last_output = last_layer.output
# Define a Callback class that stops training once accuracy reaches a threshold



class myCallback(tf.keras.callbacks.Callback):

  def on_epoch_end(self, epoch, logs={}):

    if(logs.get('acc')>0.875):

      print("\nReached 87.5% accuracy so cancelling training!")

      self.model.stop_training = True
from tensorflow.keras.optimizers import RMSprop



# Add layers after the picked last_layer

# Note that the layers below are trainable. 

# On the other hand, the layers up to the last_output is fronzen and non-trainable.



# Flatten the output layer to 1 dimension

x = layers.Flatten()(last_output)

# Add a fully connected layer with 1,024 hidden units and ReLU activation

x = layers.Dense(1024, activation='relu')(x)

# Add a dropout rate of 0.2

x = layers.Dropout(0.2)(x)                  

# Add a final sigmoid layer for binary classification

x = layers.Dense(1, activation='sigmoid')(x)           



# instantiate a new model 

model = Model(pre_trained_model.input ,x) 



# Compile a model

model.compile(optimizer = RMSprop(lr=0.0001),  # We use RMSprop in this case

              loss = 'binary_crossentropy',  # cats vs. dogs is a binary problem

              metrics = ['acc']) # 'acc' stands for accuracy



# See the summary of the model 

# model.summary()
print(os.listdir("../input/dogs-vs-cats"))
# Delete the directory we will create if it already exists

try:

    shutil.rmtree("/tmp/cat_or_dog/")    

except:

    pass



# Create a new directory for the dataset

try:

    os.mkdir("/tmp/cat_or_dog/")

except:

    pass
# Unzip　the training data

local_zip = '../input/dogs-vs-cats/train.zip'

zip_ref = zipfile.ZipFile(local_zip, 'r')

zip_ref.extractall('/tmp/cat_or_dog')

zip_ref.close()
os.listdir('/tmp/cat_or_dog')
# Login to kaggle 

# Join the competition where the data are stored

# Go to My account and click "Create new API Token" 

# A json file will be downloaded. It can be opened in any text file

# Set your kaggle username and key in the dictionary below 





# api_token = {"username":"your_kaggle_username","key":"your_kaggle_key"}
# Create a .kaggle　directory in the root directory 





# try:

#     os.mkdir("/root/.kaggle/")

# except:

#     pass
# Create a json file which stores the api keys





# with open('/root/.kaggle/kaggle.json', 'w') as file:

#     json.dump(api_token, file)
# Set owner permission to the json file 



# ! chmod 600 /root/.kaggle/kaggle.json
# Delete the directory we will create if it already exists



# try:

#     shutil.rmtree("/tmp/cat_or_dog/")    

# except:

#     pass



# # Create a new directory for the dataset

# try:

#     os.mkdir("/tmp/cat_or_dog/")

# except:

#     pass
# Go to the data tab in the kaggle competition and copy the command  

# After -p is the directory to store the data



# ! kaggle competitions download -c dogs-vs-cats -p /tmp/cat_or_dog/
# Unzip　the training data





# local_zip = '/tmp/cat_or_dog/train.zip'

# zip_ref = zipfile.ZipFile(local_zip, 'r')

# zip_ref.extractall('/tmp/cat_or_dog')

# zip_ref.close()
# Cats and dogs images are stored in the same dirctory

print(os.listdir('/tmp/cat_or_dog'))

print(os.listdir('/tmp/cat_or_dog/train')[:10])

files_list = os.listdir('/tmp/cat_or_dog/train')
# Create directories for training and validation



base_train_dir = '/tmp/cat_or_dog/training/'

base_val_dir = '/tmp/cat_or_dog/validation/'



# Delete the directory we will create if it already exists

try:

    shutil.rmtree(base_train_dir)    

except:

    pass



try:

    shutil.rmtree(base_val_dir)    

except:

    pass



# Directory with our training cat/dog pictures

train_cats_dir = os.path.join(base_train_dir, 'cats/')

train_dogs_dir = os.path.join(base_train_dir, 'dogs/')



# Directory with our validation cat/dog pictures

val_cats_dir = os.path.join(base_val_dir, 'cats/')

val_dogs_dir = os.path.join(base_val_dir, 'dogs/')





try:

    os.mkdir(base_train_dir)

    os.mkdir(base_val_dir)

    os.mkdir(train_cats_dir)

    os.mkdir(train_dogs_dir)

    os.mkdir(val_cats_dir)

    os.mkdir(val_dogs_dir)  

except:

    pass
# When using .flow_from_directory in the generator,  the "training" directory should have sub-directories for the classes, "dogs" and "cats"

# Another option is to use .flow_from_dataframe. In this case, a dataframe with filename and corresponding categories is required. 

print(os.listdir(base_train_dir))



# Same for the "validation" directory 

print(os.listdir(base_val_dir))
# Shuffle the files before splting into training set and test set

random.seed(0)

random_files_list = random.sample(files_list,len(files_list))

print(random_files_list[:10])

print(len(files_list))

print(len(random_files_list))
# Deleat empty files

# In this dataset, there is no empty files

SOURCE = '/tmp/cat_or_dog/train/'



random_files_clean_list = []



for filename in random_files_list:

    file = os.path.join(SOURCE + filename)

    if os.path.getsize(file) > 0:

        random_files_clean_list.append(filename)

    else:

        print(filename + " is zero length, so ignoring.")



# print(random_files_clean_list)

print(len(random_files_list))

print(len(random_files_clean_list))
# Split the files into training set and validation set



# Set the ratio for validation set

val_ratio = 0.3 



training_list = random_files_clean_list[:int(len(random_files_clean_list)*(1 - val_ratio))]

val_list = random_files_clean_list[-int(len(random_files_clean_list)*(val_ratio)):]



print(len(training_list))

print(len(val_list))

print(len(training_list) + len(val_list))
# Create the training set



SOURCE = '/tmp/cat_or_dog/train/' # the current directory where files are stored



for filename in training_list:

    this_file = os.path.join(SOURCE + filename)

    cat_destination = os.path.join(train_cats_dir + filename)

    dog_destination = os.path.join(train_dogs_dir + filename)

    if 'cat' in filename:

        copyfile(this_file, cat_destination)

    elif 'dog' in filename:

        copyfile(this_file, dog_destination)
train_dog_fnames = os.listdir(train_dogs_dir)

train_cat_fnames = os.listdir(train_cats_dir)





print(os.listdir(train_cats_dir)[:5])

print(os.listdir(train_dogs_dir)[:5])

print(len(os.listdir(train_cats_dir)))

print(len(os.listdir(train_dogs_dir)))
# Create the validation set



SOURCE = '/tmp/cat_or_dog/train/' # the current directory where files are stored



for filename in val_list:

    this_file = os.path.join(SOURCE + filename)

    cat_destination = os.path.join(val_cats_dir + filename)

    dog_destination = os.path.join(val_dogs_dir + filename)



    

    if 'cat' in filename:

        copyfile(this_file, cat_destination)

    elif 'dog' in filename:

        copyfile(this_file, dog_destination)
# print(os.listdir(val_cats_dir))

# print(os.listdir(val_dogs_dir))

print(len(os.listdir(val_cats_dir)))

print(len(os.listdir(val_dogs_dir)))
%matplotlib inline



import matplotlib.image as mpimg

import matplotlib.pyplot as plt



# Parameters for our graph; we'll output images in a 4x4 configuration

nrows = 4

ncols = 4



pic_index = 0 # Index for iterating over images
# Set up matplotlib fig, and size it to fit 4x4 pics

# The 8 images on the upper half are dogs, others are cats 

fig = plt.gcf()

fig.set_size_inches(ncols*3, nrows*3)



pic_index+=8



next_cat_pix = [os.path.join(train_cats_dir, fname) 

                for fname in train_cat_fnames[ pic_index-8:pic_index] 

               ]



next_dog_pix = [os.path.join(train_dogs_dir, fname) 

                for fname in train_dog_fnames[ pic_index-8:pic_index]

               ]



for i, img_path in enumerate(next_cat_pix+next_dog_pix):

  # Set up subplot; subplot indices start at 1

  sp = plt.subplot(nrows, ncols, i + 1)

  sp.axis('Off') # don't show axes (or gridlines)



  img = mpimg.imread(img_path)

  plt.imshow(img)



plt.show()
# Add our data-augmentation parameters to ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255., # rescale the images within 0-1. This yield better results when handling images. 

                                   rotation_range = 40,

                                   width_shift_range = 0.2,

                                   height_shift_range = 0.2,

                                   shear_range = 0.2,

                                   zoom_range = 0.2,

                                   horizontal_flip = True)



# Note that the validation data should not be augmented

test_datagen = ImageDataGenerator(rescale = 1.0 / 255.0)



# Flow training images in batches of 20 using train_datagen generator

train_generator = train_datagen.flow_from_directory(base_train_dir,

                                                    batch_size = 20,

                                                    class_mode = 'binary',

                                                    target_size = (150, 150))     



# Flow validation images in batches of 20 using test_datagen generator

validation_generator =  test_datagen.flow_from_directory(base_val_dir,

                                                         batch_size = 20,

                                                         class_mode = 'binary',

                                                         target_size = (150,150))
# See the indices of the classes

# In keras ImageDataGenerator class, classes are indexed in alphabetical order

train_generator.class_indices
# Fit the model the training data

# Take the log of the training in history

callbacks = myCallback()

history = model.fit_generator(

    generator = train_generator, # feed the training data via the generator

    steps_per_epoch = 8, # this is the batch size. parameters are updated per this batch size

    epochs = 100, # number of cycles. In one epoch, the whole dataset is used once.

    verbose = 2, # print out the logs

    callbacks = [callbacks], # use callbacks we set before

    validation_data = validation_generator, # feed the validation data via the generator

    validation_steps = 50 

)
# Plot the accuracy history 

import matplotlib.pyplot as plt

acc = history.history['acc']

val_acc = history.history['val_acc']

loss = history.history['loss']

val_loss = history.history['val_loss']



epochs = range(len(acc))



plt.plot(epochs, acc, 'r', label='Training accuracy')

plt.plot(epochs, val_acc, 'b', label='Validation accuracy')

plt.title('Training and validation accuracy')

plt.legend(loc=0)

plt.figure()





plt.show()
# Inside the kaggle kernel

# Unzip　the test data

local_zip = '../input/dogs-vs-cats/test1.zip'

zip_ref = zipfile.ZipFile(local_zip, 'r')

zip_ref.extractall('/tmp/cat_or_dog')

zip_ref.close()
# Outside the kaggle kernel

# Unzip　the test data

# local_zip = '/tmp/cat_or_dog/test1.zip'

# zip_ref = zipfile.ZipFile(local_zip, 'r')

# zip_ref.extractall('/tmp/cat_or_dog')

# zip_ref.close()
base_test_dir = '/tmp/cat_or_dog/test1/'



# print(os.listdir(base_test_dir))
# To use .flow_from_dataframe. in the generator, we have to create a dataframe  

test_filenames = os.listdir(base_test_dir)

test_df = pd.DataFrame({

    'filename': test_filenames

})

nb_samples = test_df.shape[0]
# Create a generator for test set

test_gen = ImageDataGenerator(rescale=1./255.0)

batch_size=20



test_generator = test_gen.flow_from_dataframe(

    test_df, 

    base_test_dir, 

    x_col='filename',

    y_col=None,

    class_mode=None,

    batch_size=batch_size,

    target_size=(150, 150),

    shuffle=False

)
# Predict the test set

predict = model.predict_generator(test_generator, steps=np.ceil(nb_samples/batch_size))

threshold = 0.5

test_df['category'] = np.where(predict > threshold, 1,0) # we can also 
# See sample results, # 1 = dog, 0 = cat

sample_test = test_df.head(18)

sample_test.head()

plt.figure(figsize=(9, 18))

for index, row in sample_test.iterrows():

    filename = row['filename']

    category = row['category']

    img = load_img(base_test_dir+filename, target_size=(150,150))

    plt.subplot(6, 3, index+1)

    plt.imshow(img)

    plt.xlabel(filename + '(' + "{}".format(category) + ')' ) 

plt.tight_layout()

plt.show()
submission_df = test_df.copy()

submission_df['id'] = submission_df['filename'].str.split('.').str[0]

submission_df['label'] = submission_df['category']

submission_df.drop(['filename', 'category'], axis=1, inplace=True)

submission_df.to_csv('/tmp/cat_or_dog/submission_20191230.csv', index=False)
!  kaggle competitions submit -c dogs-vs-cats -f /tmp/cat_or_dog/ submission_20191229.csv -m "First submission"