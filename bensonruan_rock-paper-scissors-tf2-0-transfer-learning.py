import os

import tensorflow as tf

import matplotlib.pyplot as plt

import matplotlib.image as mpimage

import numpy as np

from zipfile import ZipFile

from tensorflow.keras.preprocessing import image

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.applications.vgg16 import VGG16
print(tf.__version__)
data_dir = '../input/rockpaperscissors/rps-cv-images'

rock_dir = 'rock'

paper_dir ='paper'

scissors_dir = 'scissors'



rock_files = os.listdir(os.path.join(data_dir,rock_dir))

paper_files = os.listdir(os.path.join(data_dir,paper_dir))

scissors_files = os.listdir(os.path.join(data_dir,scissors_dir))



print('total training rock images:', len(rock_files))

print('total training paper images:', len(paper_files))

print('total training scissors images:', len(scissors_files))



pic_index = 100



next_rock = [os.path.join(data_dir, rock_dir, fname) 

                for fname in rock_files[pic_index-1:pic_index]]

next_paper = [os.path.join(data_dir, paper_dir, fname) 

                for fname in paper_files[pic_index-1:pic_index]]

next_scissors = [os.path.join(data_dir, scissors_dir, fname) 

                for fname in scissors_files[pic_index-1:pic_index]]



f, axarr = plt.subplots(1,3, figsize=(30,20))

for i, img_path in enumerate(next_rock+next_paper+next_scissors):

    img = mpimage.imread(img_path)

    axarr[i].imshow(img)

    axarr[i].axis('Off')

plt.show()
img_generator = ImageDataGenerator(validation_split=0.2,

                                  rescale = 1./255,

                                  rotation_range=20,

                                  width_shift_range=0.2,

                                  height_shift_range=0.2,

                                  shear_range=0.2,

                                  zoom_range=0.2,

                                  horizontal_flip=True,

                                  vertical_flip = True

                                  )



train_generator = img_generator.flow_from_directory(

                        data_dir,

                        target_size=(224,224),

                        batch_size=32,

                        class_mode='categorical',

                        shuffle=True,

                        subset='training'

                    )



validation_generator = img_generator.flow_from_directory(

                        data_dir,

                        target_size=(224,224),

                        batch_size=32,

                        class_mode='categorical',

                        shuffle=False,

                        subset='validation'

                    )
base_model = VGG16(weights='imagenet',include_top=False,input_shape=(224,224,3))

base_model.trainable = False



model = tf.keras.models.Sequential([

    base_model,

    tf.keras.layers.GlobalAveragePooling2D(),

    tf.keras.layers.Dense(512, activation='relu'),

    tf.keras.layers.Dropout(0.5),

    tf.keras.layers.Dense(3, activation='softmax')

])



model.summary()
model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit_generator(

    train_generator,  

    validation_data  = validation_generator,

    epochs = 10, 

    verbose = 1

)
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
model.save('model.h5')