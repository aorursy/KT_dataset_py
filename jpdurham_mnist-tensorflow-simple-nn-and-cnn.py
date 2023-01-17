# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # drawing graphs and images

import tensorflow as tf # data processing, modeling

from tensorflow.keras.preprocessing.image import NumpyArrayIterator, ImageDataGenerator

from tensorflow.keras.preprocessing.image import array_to_img, img_to_array

tf.enable_eager_execution() # will reduce memory footprint when working with numpy arrays



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train_df = pd.read_csv('../input/train.csv')

test_df = pd.read_csv('../input/test.csv')
print('Train head: {}'.format(train_df.head()))

print('Test head: {}'.format(test_df.head()))

print('------')

train_shape = train_df.shape

test_shape = test_df.shape

print('Train: {}'.format(train_shape))

print('Test: {}'.format(test_shape))
num_training_examples = train_shape[0]

num_test_examples = test_shape[0]



# Create a train and test dataset

# Select all rows and columns but first column, which is the label for the given image data

training_examples = tf.convert_to_tensor(train_df.iloc[:, 1:].values)

training_labels = tf.convert_to_tensor(train_df.iloc[:, :1].values)

train_dataset = tf.data.Dataset.from_tensor_slices((training_examples, training_labels))



testing_examples = tf.convert_to_tensor(test_df.values)

test_dataset = tf.data.Dataset.from_tensor_slices(testing_examples)



# Normalize the pixel values between 0 and 1 (smaller values play nice with the model we'll build)

def normalize(images, labels):

    images = tf.cast(images, tf.float32)

    images /= 255

    return images, labels



def normalize_unlabled(images):

    images = tf.cast(images, tf.float32)

    images /= 255

    return images



train_dataset = train_dataset.map(normalize)

test_dataset =  test_dataset.map(normalize_unlabled)
# We're dealing with square images, so let's calculate the size of each image

IMG_SIZE = int(np.sqrt(len(training_examples[0])))



# Take a single image and format it

for image in test_dataset.take(1):

    break

image = image.numpy().reshape((IMG_SIZE, IMG_SIZE))



# Plot the image

plt.figure()

plt.imshow(image, cmap=plt.cm.binary)

plt.colorbar()

plt.grid(False)

plt.show()
# Looks like our test dataset is formatted correctly so

# take a look at the training data

plt.figure(figsize=(10,10))

i = 0

for (image, label) in train_dataset.take(20):

    image = image.numpy().reshape((IMG_SIZE, IMG_SIZE))

    plt.subplot(5,4,i+1)

    plt.xticks([])

    plt.yticks([])

    plt.grid(False)

    plt.imshow(image, cmap=plt.cm.binary)

    plt.xlabel(label.numpy()[0], fontsize=16)

    i += 1

plt.show()
BATCH_SIZE = 32 # 32 - 512 is a good range



# Last step is creating a validation dataset

# which we can use to monitor our training progress.

# We'll use 80% of the data for training

train_split = int(0.8 * num_training_examples)

validation_dataset = train_dataset.skip(train_split).batch(BATCH_SIZE)

train_dataset = train_dataset.take(train_split).repeat().shuffle(train_split).batch(BATCH_SIZE)
flat_model = tf.keras.Sequential([

    tf.keras.layers.Dense(32, input_shape=(784,)),

    tf.keras.layers.Dense(128, activation=tf.nn.relu),

    tf.keras.layers.Dense(10,  activation=tf.nn.softmax)

])



flat_model.summary()



flat_model.compile(optimizer='adam', 

              loss='sparse_categorical_crossentropy',

              metrics=['accuracy'])



history = flat_model.fit(train_dataset,

               epochs=6,

               steps_per_epoch=np.ceil(num_training_examples/BATCH_SIZE),

               validation_data=validation_dataset,

               validation_steps=np.ceil((num_training_examples-train_split)/BATCH_SIZE))
x = test_dataset.batch(num_test_examples)

predictions = flat_model.predict(x)



def plot_image(i, predictions_array, images):

  predictions_array, img = predictions_array[i], images[i].numpy().reshape((IMG_SIZE, IMG_SIZE))

  plt.grid(False)

  plt.xticks([])

  plt.yticks([])

  

  plt.imshow(img, cmap=plt.cm.binary)



  predicted_label = np.argmax(predictions_array)

  color = 'green'

  

  plt.xlabel("{} {:2.0f}%".format(predicted_label,

                                100*np.max(predictions_array),

                                color=color))



num_rows = 5

num_cols = 5

num_images = num_rows*num_cols

plt.figure(figsize=(2*num_cols, 2*num_rows))

for i in range(num_images):

  plt.subplot(num_rows, num_cols, i+1)

  plot_image(i, predictions, testing_examples)
# Get predicted labels

predicted_labels = [np.argmax(prediction) for prediction in predictions]



# Create submission

np.savetxt('submission.csv', 

           np.c_[range(1,num_test_examples+1),predicted_labels], 

           delimiter=',', 

           header = 'ImageId,Label', 

           comments = '', 

           fmt='%d')
# Leverage TensorFlow (Keras) image pre-processing and data augmentation capabilities

image_gen_train = ImageDataGenerator(

      rescale=1./255,

      rotation_range=10,

#       width_shift_range=0.15,

#       height_shift_range=0.15,

      zoom_range=0.1,

      fill_mode='nearest',

      validation_split=0.9

)



train_imgs = np.asarray([img.reshape((IMG_SIZE, IMG_SIZE, 1)) for img in train_df.iloc[:, 1:].values])

train_labels = train_df.iloc[:, :1]

train_img_gen = NumpyArrayIterator(train_imgs, train_labels, shuffle=True, subset='training', batch_size=BATCH_SIZE, image_data_generator=image_gen_train)

val_img_gen = NumpyArrayIterator(train_imgs, train_labels, subset='validation', batch_size=BATCH_SIZE, image_data_generator=image_gen_train)



# Plot images from an array

def plot_images(images_arr):

    fig, axes = plt.subplots(1, 4, figsize=(20,20))

    axes = axes.flatten()

    for img, ax in zip( images_arr, axes):

        ax.imshow(array_to_img(img))

    plt.tight_layout()

    plt.show()



# Pull sample data from generator to show how same image may be augemented

augmented_images = [train_img_gen[0][0][0] for i in range(10)]

plot_images(augmented_images)



conv_model = tf.keras.models.Sequential([

    tf.keras.layers.Conv2D(32, (3, 3), input_shape=(28, 28, 1), activation='relu'),

    tf.keras.layers.MaxPooling2D(2, 2),

    

    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),

    tf.keras.layers.MaxPooling2D(2, 2),

    

    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),

    tf.keras.layers.MaxPooling2D(2, 2),

    

    tf.keras.layers.Dropout(0.25),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(1024, activation='relu'),

    tf.keras.layers.Dense(10, activation='softmax')

])



# Compile the model

conv_model.compile(

    optimizer='adam',

    loss='sparse_categorical_crossentropy',

    metrics=['accuracy']

)



conv_model.summary()



epochs = 3

steps_per_epoch=int(np.ceil(len(train_imgs) / float(BATCH_SIZE)))

validation_steps=int(np.ceil(len(train_imgs)*0.8) / float(BATCH_SIZE))



history = conv_model.fit_generator(

    train_img_gen,

    steps_per_epoch=steps_per_epoch,

    epochs=epochs,

    validation_data=val_img_gen,

    validation_steps=validation_steps

)
z = np.asarray([img.reshape((IMG_SIZE, IMG_SIZE, 1))/255.0 for img in test_df.values])

conv_predictions = conv_model.predict(z)



def plot_conv_image(i, predictions_array, images):

  predictions_array, img = predictions_array[i], images[i]

  plt.grid(False)

  plt.xticks([])

  plt.yticks([])

  

  plt.imshow(array_to_img(img), cmap=plt.cm.binary)



  predicted_label = np.argmax(predictions_array)

  color = 'blue'

  

  plt.xlabel("{} {:2.0f}%".format(predicted_label,

                                100*np.max(predictions_array),

                                color=color))



num_rows = 5

num_cols = 5

num_images = num_rows*num_cols

plt.figure(figsize=(2*num_cols, 2*num_rows))

for i in range(num_images):

  plt.subplot(num_rows, num_cols, i+1)

  plot_conv_image(i, conv_predictions, z)
# Get predicted labels

conv_predicted_labels = [np.argmax(prediction) for prediction in conv_predictions]



# Create submission

np.savetxt('conv_submission.csv', 

           np.c_[range(1,len(conv_predicted_labels)+1),predicted_labels], 

           delimiter=',', 

           header = 'ImageId,Label', 

           comments = '', 

           fmt='%d')