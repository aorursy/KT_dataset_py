import numpy as np

import pandas as pd 

import keras

import tensorflow as tf

from keras.preprocessing.image import ImageDataGenerator, load_img

from keras.utils import to_categorical

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

import random

import os

print(os.listdir("../input"))

# assign FAST_RUN=True to train the model with three epochs

FAST_RUN = False



# input image dimensions

IMAGE_WIDTH=128

IMAGE_HEIGHT=128

IMAGE_SIZE=(IMAGE_WIDTH, IMAGE_HEIGHT)



# red, green and blue channels

IMAGE_CHANNELS=3
# get a list of file names from the train folder

filenames = os.listdir("../input/train/train")

filenames
# an empty list which will contain all the labels of the train images.

# for example: for cat.8937.jpg- 0 will be appended to the list. for dog.695.jpg- 1 will be 

# appended to the list

categories = []

for filename in filenames:

    

    # split the filename using delimiter '.'.

    # for example, 'cat.8937.jpg' will be splitted into 'cat','8937','jpg'. we will take the

    # first string 'cat' as the category of that image

    category = filename.split('.')[0]

    

    # We will label all the images with dog photos as 1's and cat photos as 0's

    if category == 'dog':

        categories.append(1)

    else:

        categories.append(0)



# we will create a dataframe which will contain the filenames and the labels of our train set

df = pd.DataFrame({

    'filename': filenames,

    'category': categories

})
df.head()
df.tail()
# create a barplot showing the amount of cat and dog pictures

df['category'].value_counts().plot.bar()
# randomly choose an image for display

sample = random.choice(filenames)

image = load_img("../input/train/train/"+sample)

plt.imshow(image)
from keras.models import Sequential

from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation, BatchNormalization



model = Sequential()



# layer 1

model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS)))

model.add(BatchNormalization())



model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))



#layer 2

model.add(Conv2D(64, (3, 3), activation='relu'))

model.add(BatchNormalization())



model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))



#layer 3

model.add(Conv2D(128, (3, 3), activation='relu'))

model.add(BatchNormalization())



model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))



#fully connected layer

model.add(Flatten())



model.add(Dense(512, activation='relu'))

model.add(BatchNormalization())

model.add(Dropout(0.5))



model.add(Dense(2, activation='softmax')) # 2 because we have cat and dog classes
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])



model.summary()
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
earlystop = EarlyStopping(patience=10)
learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', 

                                            patience=2, 

                                            verbose=1, 

                                            factor=0.5, 

                                            min_lr=0.00001)
callbacks = [earlystop, learning_rate_reduction]
df["category"] = df["category"].replace({0: 'cat', 1: 'dog'}) 
# do a train-validation split on the whole set.

train_df, validate_df = train_test_split(df, test_size=0.20, random_state=42)



# drop the indexes of train and validation dataframe

train_df = train_df.reset_index(drop=True)

validate_df = validate_df.reset_index(drop=True)
# see the amount of cat and dog photos in the train dataframe

train_df['category'].value_counts().plot.bar()
# see the amount of cat and dog photos in the train dataframe

validate_df['category'].value_counts().plot.bar()
# get the total amount of data in train and validation set

total_train = train_df.shape[0]

total_validate = validate_df.shape[0]



# set the minibatch size to 15

batch_size=15
# ImageDataGenerator?
#  Generate batches of tensor image data with real-time data augmentation.

#  The data will be looped over (in batches)



# Degree range for random rotations is 15

# rescaling factor is 1/255, meaning that the image pixel values will be multiplied by 1/255

# Shear angle in counter-clockwise direction in degrees is 0.1

# Range for random zoom is 0.2

# Randomly flip inputs horizontally

# width_shift_range 0.1 fraction of total width

# height_shift_range 0.1 fraction of total height

train_datagen = ImageDataGenerator(

    rotation_range=15,

    rescale=1./255,

    shear_range=0.1,

    zoom_range=0.2,

    horizontal_flip=True,

    width_shift_range=0.1,

    height_shift_range=0.1

)
# train_datagen.flow_from_dataframe?
# Takes the dataframe and the path to a directory

#  and generates batches of augmented/normalized data.



# target_size: The dimensions to which all images found will be resized

train_generator = train_datagen.flow_from_dataframe(

    train_df, 

    "../input/train/train/", 

    x_col='filename',

    y_col='category',

    target_size=IMAGE_SIZE,

    class_mode='categorical',

    batch_size=batch_size

)
# we will definitely not need to zoom, shear, or any kind of bullshit to increase validation

# set but we will divide the pixel values by 255



validation_datagen = ImageDataGenerator(rescale=1./255)

validation_generator = validation_datagen.flow_from_dataframe(

    validate_df, 

    "../input/train/train/",

    x_col='filename',

    y_col='category',

    target_size=IMAGE_SIZE,

    class_mode='categorical',

    batch_size=batch_size

)
# here we will create a dataframe with one row from the training dataframe for demonstrating

# how the datagenerator works

example_df = train_df.sample(n=1).reset_index(drop=True)

example_generator = train_datagen.flow_from_dataframe(

    example_df, 

    "../input/train/train/", 

    x_col='filename',

    y_col='category',

    target_size=IMAGE_SIZE,

    class_mode='categorical'

)
# we will generate 15 random image from our example data generator and show them using

# matplotlib

plt.figure(figsize=(12, 12))

for i in range(0, 15):

    plt.subplot(5, 3, i+1)

    #on each iteration of the for loop, the generator returns the same amount of random 

    #images as the original dataframe on which the generator was created

    for X_batch, Y_batch in example_generator:

        #get the first image of the generated batch

        image = X_batch[0]

        plt.imshow(image)

        break

plt.tight_layout()

plt.show()
if(os.path.isfile('saved_model/history.csv') and pd.read_csv('saved_model/history.csv').shape[0]>0):

    iteration_to_be_loaded=pd.read_csv('saved_model/history.csv').shape[0]-1

    model.load_weights("saved_model/model_"+str(iteration_to_be_loaded)+".h5")

    print("saved_model/model_"+str(iteration_to_be_loaded)+".h5"+" loaded!")
epochs=3 if FAST_RUN else 50

history = model.fit(

    train_generator, 

    epochs=epochs,

    validation_data=validation_generator,

    validation_steps=total_validate//batch_size,

    steps_per_epoch=total_train//batch_size,

    callbacks=callbacks

)
# assert(False)
model.save_weights("saved_model/model.h5")



import pickle

with open('saved_model/history.pickle', 'wb') as f:

    pickle.dump(history, f)
model.load_weights("saved_model/model.h5")
import pickle

with open('saved_model/history.pickle','rb') as f:

    history = pickle.load(f)
# create a figure with two subplots in 2 rows and one column

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))



#draw losses on the first subplot

# plot the training loss

ax1.plot(history.history['loss'], color='b', label="Training loss")

#plot the validation loss

ax1.plot(history.history['val_loss'], color='r', label="validation loss")



ax1.set_xticks(np.arange(1, epochs, 1))

ax1.set_yticks(np.arange(0, 1, 0.1))



#draw accuracy on the second subplot

# training accuracy

ax2.plot(history.history['accuracy'], color='b', label="Training accuracy")

# validation accuracy

ax2.plot(history.history['val_accuracy'], color='r',label="Validation accuracy")



ax2.set_xticks(np.arange(1, epochs, 1))



legend = plt.legend(loc='best', shadow=True)

plt.tight_layout()

plt.show()
test_filenames = os.listdir("../input/test1/test1")

test_df = pd.DataFrame({

    'filename': test_filenames

})

nb_samples = test_df.shape[0]
test_gen = ImageDataGenerator(rescale=1./255)



#remember to not shuffle the test set

test_generator = test_gen.flow_from_dataframe(

    test_df, 

    "../input/test1/test1/", 

    x_col='filename',

    y_col=None,

    class_mode=None,

    target_size=IMAGE_SIZE,

    batch_size=batch_size,

    shuffle=False

)
predict = model.predict_generator(test_generator, steps=np.ceil(nb_samples/batch_size))
predict
test_df['category'] = np.argmax(predict, axis=-1)
test_df['category']
label_map = dict((v,k) for k,v in train_generator.class_indices.items())

test_df['category'] = test_df['category'].replace(label_map)
test_df['category']
test_df['category'] = test_df['category'].replace({ 'dog': 1, 'cat': 0 })
test_df['category'].value_counts().plot.bar()
sample_test = test_df.head(18)

sample_test.head(18)
plt.figure(figsize=(12, 24))



# iterrows() will return index and each row of a dataframe

for index, row in sample_test.iterrows():

    filename = row['filename']

    category = row['category']

    img = load_img("../input/test1/test1/"+filename, target_size=IMAGE_SIZE)

    plt.subplot(6, 3, index+1)

    plt.imshow(img)

    plt.xlabel(filename + '(' + "{}".format(category) + ')' )

plt.tight_layout()

plt.show()
submission_df = test_df.copy()

submission_df['id'] = submission_df['filename'].str.split('.').str[0]

submission_df['label'] = submission_df['category']

submission_df.drop(['filename', 'category'], axis=1, inplace=True)

# I added this line

submission_df=submission_df.astype({'id': 'int32'})

submission_df=submission_df.sort_values('id',ascending=True)



submission_df.to_csv('submission.csv', index=False)