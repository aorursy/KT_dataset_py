# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow as tf

from sklearn.utils import shuffle

from tensorflow.keras.preprocessing.image import ImageDataGenerator



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train_data = pd.read_csv('/kaggle/input/digit-recognizer/train.csv', header=0)

test_data = pd.read_csv('/kaggle/input/digit-recognizer/test.csv', header=0)
print(train_data.shape)

print(test_data.shape)

print(len(train_data))
train_data = shuffle(train_data)

split_index = round(1*len(train_data))

training_data = train_data.iloc[:split_index,:]

#validation_data = train_data.iloc[split_index:,:]
print(training_data.shape)

#print(validation_data.shape)
def convert_to_images(data):

    images = []

    labels = []

    

    for row in range(len(data)):

        temp_images = np.array_split(data.iloc[row,1:], 28)

        temp_labels = data.iloc[row,0]

        images.append(temp_images)

        labels.append(temp_labels)

    images = np.array(images).astype('float')

    labels = np.array(labels).astype('float')

    return images, labels



train_images, train_labels = convert_to_images(training_data)

#validation_images, validation_labels = convert_to_images(validation_data)
train_images = np.expand_dims(train_images, axis = 3)

#validation_images = np.expand_dims(validation_images, axis = 3)
test_images = []

for i in range(len(test_data)):

    temp_image = np.array_split(test_data.iloc[i,:], 28)

    test_images.append(temp_image)

test_images = np.array(test_images).astype('float')

test_images = np.expand_dims(test_images, axis = 3)
print(train_images.shape)

print(train_labels.shape)

print(test_images.shape)

#print(validation_images.shape)
train_datagen = ImageDataGenerator(

    rescale = 1./255.,

    rotation_range = 0.2,

    width_shift_range = 0.2,

    height_shift_range = 0.2,

    shear_range = 0.2,

    zoom_range=0.2,

    horizontal_flip=True,

    fill_mode='nearest'

)



#validation_datagen = ImageDataGenerator(

    #rescale=1. / 255)

    

train_gen = train_datagen.flow(

    train_images,

    train_labels,

    batch_size = 168

)



# validation_gen = validation_datagen.flow(

#     validation_images,

#     validation_labels,

#     batch_size = 42

# )

    

# Keep These

print(train_images.shape)

#print(validation_images.shape)

    
model = tf.keras.models.Sequential([

    tf.keras.layers.Conv2D(32,(3,3), activation = 'relu', input_shape = (28,28,1)),

    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(64,(3,3), activation = 'relu'),

    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Dropout(0.20),

    tf.keras.layers.Conv2D(128,(3,3), activation = 'relu'),

    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(512, activation='relu'),

    tf.keras.layers.Dense(10, activation='softmax')

])



model.compile(loss = 'sparse_categorical_crossentropy' , optimizer = 'rmsprop', metrics = ['acc'])



model.summary()
history = model.fit_generator(

    train_gen,

    epochs = 50,

    verbose = 2#,

    #validation_data =  validation_gen

)
predictions = model.predict(test_images)
submissions=pd.DataFrame({"ImageId": list(range(1,len(predictions)+1)),

                         "Label": np.argmax(predictions, axis=1)})

submissions.to_csv("my_submissions_augmentation.csv", index=False, header=True)
# Plot the chart for accuracy and loss on both training and validation

%matplotlib inline

import matplotlib.pyplot as plt



acc = history.history['acc']

#val_acc = history.history['val_acc']

loss = history.history['loss']

#val_loss = history.history['val_loss']



epochs = range(len(acc))



plt.plot(epochs, acc, 'r', label='Training accuracy')

#plt.plot(epochs, val_acc, 'b', label='Validation accuracy')

plt.title('Training and validation accuracy')

plt.legend()

plt.figure()



plt.plot(epochs, loss, 'r', label='Training Loss')

#plt.plot(epochs, val_loss, 'b', label='Validation Loss')

plt.title('Training and validation loss')

plt.legend()



plt.show()