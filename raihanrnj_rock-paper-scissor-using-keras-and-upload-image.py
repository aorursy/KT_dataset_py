import tensorflow as tf

import matplotlib.pyplot as plt

import matplotlib.image as mpimage

import numpy as np



from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.preprocessing import image

from tensorflow.keras.applications.vgg16 import VGG16
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


dataset_url = '/kaggle/input/rockpaperscissors/rps-cv-images'

rock_url = 'rock'

paper_url ='paper'

scissors_url = 'scissors'



rock_results= os.listdir(os.path.join(dataset_url,rock_url))

paper_results = os.listdir(os.path.join(dataset_url,paper_url))

scissors_results = os.listdir(os.path.join(dataset_url,scissors_url))



print('Terdapat :', len(rock_results), 'Gambar Batu dalam Dataset')

print('Terdapat :', len(paper_results), 'Gambar Kertas dalam Dataset')

print('Terdapat :', len(scissors_results), 'Gambar Gunting dalam Dataset')
data_index = 100



next_rock = [os.path.join(dataset_url, rock_url, fname) 

                for fname in rock_results[data_index-1:data_index]]

next_paper = [os.path.join(dataset_url, paper_url, fname) 

                for fname in paper_results[data_index-1:data_index]]

next_scissors = [os.path.join(dataset_url, scissors_url, fname) 

                for fname in scissors_results[data_index-1:data_index]]



array,f = plt.subplots(1,3, figsize=(15,10))

for iter, img_path in enumerate(next_rock+next_paper+next_scissors):

    pict = mpimage.imread(img_path)

    array[iter].imshow(pict)

    array[iter].axis('Off')

plt.show()
image_gen     = ImageDataGenerator(validation_split=0.2,

                                  rescale = 1./255,

                                  rotation_range=20,

                                  width_shift_range=0.2,

                                  height_shift_range=0.2,

                                  shear_range=0.2,

                                  zoom_range=0.2,

                                  horizontal_flip=True,

                                  vertical_flip = True

                                  )

train_data_gen = image_gen.flow_from_directory(dataset_url,

                                              target_size=(224,224),

                                              class_mode='categorical',

                                              batch_size=32,

                                              shuffle=True,

                                              subset='training'

                                               )



validation_data_gen = image_gen.flow_from_directory(dataset_url,

                                                    target_size=(224,224),

                                                    class_mode='categorical',

                                                    batch_size=32,

                                                    shuffle=False,

                                                    subset='validation'

                                                    )
basic_cls = VGG16(weights='imagenet',include_top=False,input_shape=(224,224,3))

basic_cls.trainable = False



cls = tf.keras.models.Sequential([

    basic_cls,

    tf.keras.layers.GlobalAveragePooling2D(),

    tf.keras.layers.Dense(512, activation='relu'),

    tf.keras.layers.Dropout(0.5), #untuk mengurangi overfitting

    tf.keras.layers.Dense(3, activation='softmax')

])



cls.summary()
cls.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model_history = cls.fit_generator(

    train_data_gen,  

    validation_data  = validation_data_gen,

    epochs = 10, 

    verbose = 1

)
get_acc = model_history.history['accuracy']

value_acc = model_history.history['val_accuracy']

get_loss = model_history.history['loss']

value_loss = model_history.history['val_loss']



epochs = range(len(get_acc))

plt.plot(epochs, get_acc, 'r', label='Accuracy of Training data')

plt.plot(epochs, value_acc, 'b', label='Accuracy of Validation data')

plt.title('Training vs validation accuracy')

plt.legend(loc=0)

plt.figure()





plt.show()


import keras

from google.colab import files

from keras.preprocessing import image

import matplotlib.pyplot as plt

import matplotlib.image as mpimg

import matplotlib.pyplot as plt

import matplotlib.image as mpimg

from keras.utils.np_utils import to_categorical

%matplotlib inline



get_upload = files.upload()



for iter_up in get_upload.keys():



  #Mari lakukan prediksi terhadap gambar

  path = iter_up

  pic = image.load_img(path, target_size=(150,150))

  imgplot = plt.imshow(pic)

  x = image.img_to_array(pic)

  x = np.expand_dims(x, axis=0)



  images = np.vstack([x])

  classes = cls.predict(images, batch_size=10)

  y_classes = int(classes.argmax(axis=-1))

  print("Masuk dalam Class ", y_classes,", yaitu : ")

  if (y_classes==0):

    print("---Kertas---")

  elif (y_classes==1):

    print("---Batu---")

  elif (y_classes==2):

    print("---Gunting---")

    

  print(iter_up)

  