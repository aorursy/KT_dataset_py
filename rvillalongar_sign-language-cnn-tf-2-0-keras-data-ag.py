import tensorflow as tf

import os

import numpy as np

from tensorflow.keras.preprocessing.image import ImageDataGenerator

import pandas as pd



print(tf.__version__)

print(os.listdir('../input/'))
name_file_train = 'sign_mnist_train.csv'

name_file_test ='sign_mnist_test.csv'

directory = '../input/sign-language-mnist/'



file_train = os.path.join(directory,name_file_train)

file_test = os.path.join(directory, name_file_test)



print(directory)

print(file_train)

print(file_test)
def get_data(filename):

    df = pd.read_csv(filename, header=0)

    labels= np.array(df.iloc[:,0].values)

    imgs = df.iloc[:,1:].values

    data = []

    for img in imgs:

        tmp =np.array( np.array_split(img,28))

        data.append(tmp)

    

    data = np.array(data).astype('float')    

    return data, labels



training_images, training_labels = get_data(file_train)

testing_images, testing_labels = get_data(file_test)



print(training_images.shape)

print(training_labels.shape)

print(testing_images.shape)

print(testing_labels.shape)
training_images = np.expand_dims(training_images,axis=3)

testing_images = np.expand_dims(testing_images,axis=3)



print(training_images.shape)

print(testing_images.shape)
# Create an ImageDataGenerator 

# and do Image Augmentation

train_datagen = ImageDataGenerator(

    rescale=1.0/255.0,

    rotation_range=40,

    width_shift_range=0.2,

    height_shift_range=0.2,

    shear_range=0.2,

    zoom_range=0.2, 

    horizontal_flip=True,

    fill_mode='nearest'

    )



validation_datagen = ImageDataGenerator(

    rescale=1.0/255.0)
model = tf.keras.models.Sequential([

    tf.keras.layers.Conv2D(32,(3,3), activation='relu', input_shape=(28,28,1)),

    tf.keras.layers.MaxPool2D(2,2),

    tf.keras.layers.Conv2D(64,(3,3), activation='relu'),

    tf.keras.layers.MaxPool2D(2,2),

    tf.keras.layers.Conv2D(128,(3,3), activation='relu'),

    tf.keras.layers.MaxPool2D(2,2),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Dense(1024, activation='relu'),

    tf.keras.layers.Dense(26, activation='softmax')    

])



model.summary()
# Compile Model. 

model.compile(optimizer='Adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])



# Train the Model

history = model.fit_generator(

    train_datagen.flow(training_images,training_labels,batch_size=32),

    steps_per_epoch=len(training_images)/32,

    epochs=50,

    validation_data=validation_datagen.flow(testing_images, testing_labels, batch_size=32),

    validation_steps=len(testing_images)/32

                             )



model.evaluate(testing_images, testing_labels)
import matplotlib.pyplot as plt

acc = history.history['accuracy']

val_acc = history.history['val_accuracy']

loss = history.history['loss']

val_loss = history.history['val_loss']



epochs = range(len(acc))



plt.plot(epochs, acc, 'r', label='Training accuracy')

plt.plot(epochs, val_acc, 'b', label='Validation accuracy')

plt.title('Training and validation accuracy')

plt.legend()

plt.figure()



plt.plot(epochs, loss, 'r', label='Training Loss')

plt.plot(epochs, val_loss, 'b', label='Validation Loss')

plt.title('Training and validation loss')

plt.legend()



plt.show()