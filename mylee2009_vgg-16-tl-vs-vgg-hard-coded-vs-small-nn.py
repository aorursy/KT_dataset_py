import numpy as np 

import pandas as pd 

import tensorflow as tf

from tensorflow import keras

from keras.preprocessing.image import ImageDataGenerator, load_img

import os

import random

import matplotlib.pyplot as plt

from IPython.display import Image

from tensorflow.keras.optimizers import RMSprop

from keras.applications import VGG16

from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation,GlobalMaxPooling2D

from keras.models import Model

from keras import optimizers
import zipfile

with zipfile.ZipFile("../input/dogs-vs-cats/train.zip","r") as zip_ref:

    zip_ref.extractall("train")



with zipfile.ZipFile("../input/dogs-vs-cats/test1.zip","r") as zip_ref:

    zip_ref.extractall("test1")
train_directory = "train/train/"

test_directory  = "test1/test1/"



# See sample image

filenames = os.listdir(train_directory)

sample = random.choice(filenames)

print(sample)

image = load_img(train_directory + sample)

plt.imshow(image)
filenames[:5]
# 8000 train samples

# 1600 validation samples

import shutil



source_dir = 'train/'

def copy_files(prefix_str, range_start, range_end, target_dir):

    image_paths = []

    for i in range(range_start, range_end):

        image_path = os.path.join(source_dir,'train', prefix_str + '.'+ str(i)+ '.jpg')

        image_paths.append(image_path)

    dest_dir = os.path.join( 'data', target_dir, prefix_str)

    os.makedirs(dest_dir)



    for image_path in image_paths:

        shutil.copy(image_path,  dest_dir)



copy_files('dog', 0, 4000, 'train')

copy_files('cat', 0, 4000, 'train')

copy_files('dog', 4000, 4800,'validation')

copy_files('cat', 4000, 4800, 'validation')
print(len(os.listdir('data/train/cat')))

print(len(os.listdir('data/train/dog')))

print(len(os.listdir('data/validation/cat')))

print(len(os.listdir('data/validation/dog')))
#remove train folder to free up space

if  os.path.exists('train'):

    #os.removedirs("train")

    shutil.rmtree("train") 
# adding the VGG architecture below

Image("../input/vgg-architecture-image/VGG.png")
## p.s.: VGG model is too big for kaggle to run (out of memory), removed a few FC layers and reduced the one dense layer size from 4096 to 1024



model_vgg = tf.keras.models.Sequential([

           tf.keras.layers.Conv2D(filters = 64, kernel_size = (3,3), padding = 'same', activation = 'relu', input_shape=(128,128,3)),

           tf.keras.layers.Conv2D(filters = 64, kernel_size = (3,3), padding = 'same', activation = 'relu'),

           tf.keras.layers.MaxPooling2D(pool_size=2, strides=None, padding='valid'),

           tf.keras.layers.Conv2D(filters = 128, kernel_size = (3,3), padding = 'same', activation = 'relu'),

           tf.keras.layers.Conv2D(filters = 128, kernel_size = (3,3), padding = 'same', activation = 'relu'),

           tf.keras.layers.MaxPooling2D(pool_size=2, strides=None, padding='valid'),

           tf.keras.layers.Conv2D(filters = 256, kernel_size = (3,3), padding = 'same', activation = 'relu'),

           tf.keras.layers.Conv2D(filters = 256, kernel_size = (3,3), padding = 'same', activation = 'relu'),

           tf.keras.layers.Conv2D(filters = 256, kernel_size = (3,3), padding = 'same', activation = 'relu'),

           tf.keras.layers.MaxPooling2D(pool_size=2, strides=None, padding='valid'),

           tf.keras.layers.Conv2D(filters = 512, kernel_size = (3,3), padding = 'same', activation = 'relu'),

           tf.keras.layers.Conv2D(filters = 512, kernel_size = (3,3), padding = 'same', activation = 'relu'),

           tf.keras.layers.Conv2D(filters = 512, kernel_size = (3,3), padding = 'same', activation = 'relu'),

           tf.keras.layers.MaxPooling2D(pool_size=2, strides=None, padding='valid'),

           tf.keras.layers.Conv2D(filters = 512, kernel_size = (3,3), padding = 'same', activation = 'relu'),

           tf.keras.layers.Conv2D(filters = 512, kernel_size = (3,3), padding = 'same', activation = 'relu'),

           tf.keras.layers.Conv2D(filters = 512, kernel_size = (3,3), padding = 'same', activation = 'relu'),

           tf.keras.layers.MaxPooling2D(pool_size=2, strides=None, padding='valid'),

           tf.keras.layers.Flatten(),

           tf.keras.layers.Dense(1024, activation='relu'),

#            tf.keras.layers.Dense(4096, activation='relu'),

#            tf.keras.layers.Dense(4096, activation='relu'),

           tf.keras.layers.Dense(1, activation='sigmoid')

])



model_vgg.compile(optimizer=RMSprop(lr=0.001), loss='binary_crossentropy', metrics=['acc'])



model_vgg.summary()
train_data_dir = 'data/train'

validation_data_dir = 'data/validation'



train_datagen = ImageDataGenerator(rescale=1./255,

      rotation_range=40,

      width_shift_range=0.2,

      height_shift_range=0.2,

      shear_range=0.2,

      zoom_range=0.2,

      horizontal_flip=True,

      fill_mode='nearest')

train_generator = train_datagen.flow_from_directory(train_data_dir,

                                                    batch_size=32,

                                                    class_mode='binary',

                                                    target_size=(128,128))



validation_datagen = ImageDataGenerator(rescale=1./255,

      rotation_range=40,

      width_shift_range=0.2,

      height_shift_range=0.2,

      shear_range=0.2,

      zoom_range=0.2,

      horizontal_flip=True,

      fill_mode='nearest')

validation_generator = validation_datagen.flow_from_directory(validation_data_dir,

                                                              batch_size=32,

                                                              class_mode='binary',

                                                              target_size=(128,128))



history = model_vgg.fit_generator(train_generator,

                              epochs=15,

                              verbose=1,

                              validation_data=validation_generator)
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

plt.figure()
model_nn = tf.keras.models.Sequential([

    tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(128, 128, 3)),

    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),

    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),

    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(512, activation='relu'),

    tf.keras.layers.Dense(1, activation='sigmoid')

])



model_nn.compile(optimizer=RMSprop(lr=0.001), loss='binary_crossentropy', metrics=['acc'])



history = model_nn.fit_generator(train_generator,

                              epochs=15,

                              verbose=1,

                              validation_data=validation_generator)
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

# Plot training and validation accuracy per epoch-++2-

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

plt.figure()
model_vgg_pretrained = VGG16(input_shape=(128, 128, 3), include_top=False, weights="imagenet")

model_vgg_pretrained.summary()
#freeze all layers

for layer in model_vgg_pretrained.layers[:15]:

    layer.trainable = False



for layer in model_vgg_pretrained.layers[15:]:

    layer.trainable = True

    

last_layer = model_vgg_pretrained.get_layer('block5_pool')

last_output = last_layer.output
# Flatten the output layer to 1 dimension

x = GlobalMaxPooling2D()(last_output)

# Add a fully connected layer with 512 hidden units and ReLU activation

x = Dense(512, activation='relu')(x)

# Add a dropout rate of 0.5

x = Dropout(0.5)(x)

# Add a final sigmoid layer for classification

x = Dense(1, activation='sigmoid')(x)



model_vgg_pretrained = Model(model_vgg_pretrained.input, x)



model_vgg_pretrained.compile(loss='binary_crossentropy',

              optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),

              metrics=['accuracy'])



model_vgg_pretrained.summary()
model_vgg_pretrained.compile(optimizer=RMSprop(lr=0.001), loss='binary_crossentropy', metrics=['acc'])



history = model_vgg_pretrained.fit_generator(train_generator,

                              epochs=15,

                              verbose=1,

                              validation_data=validation_generator)
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

# Plot training and validation accuracy per epoch-++2-

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

plt.figure()
test_filenames = os.listdir("test1/test1")

test_df = pd.DataFrame({

    'filename': test_filenames

})

nb_samples = test_df.shape[0]
test_gen = ImageDataGenerator(rescale=1./255)

test_generator = test_gen.flow_from_dataframe(

    test_df, 

    "test1/test1", 

    batch_size=32,

    class_mode=None,

    target_size=(128,128)

)
predict = model_vgg_pretrained.predict_generator(test_generator, steps=np.ceil(nb_samples/32))

threshold = 0.5

test_df['category'] = np.where(predict > threshold, 1,0)
sample_test = test_df.sample(n=9).reset_index()

sample_test.head()

plt.figure(figsize=(12, 12))

for index, row in sample_test.iterrows():

    filename = row['filename']

    category = row['category']

    img = load_img("test1/test1/"+filename, target_size=(256, 256))

    plt.subplot(3, 3, index+1)

    plt.imshow(img)

    plt.xlabel(filename + '(' + "{}".format(category) + ')')

plt.tight_layout()

plt.show()
import seaborn as sns

submission_df = test_df.copy()

submission_df['id'] = submission_df['filename'].str.split('.').str[0]

submission_df['label'] = submission_df['category']

submission_df.drop(['filename', 'category'], axis=1, inplace=True)

submission_df.to_csv('submission_20200202.csv', index=False)



plt.figure(figsize=(10,5))

sns.countplot(submission_df['label'])

plt.title("(Test data)")