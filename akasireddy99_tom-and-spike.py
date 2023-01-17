import os

import tensorflow as tf

from tensorflow import keras
import matplotlib.pyplot as plt



plt.rcParams['figure.figsize'] = (16, 10)
!unzip /kaggle/input/dogs-vs-cats/train.zip -d /kaggle/working/

!unzip /kaggle/input/dogs-vs-cats/test1.zip -d /kaggle/working/

!ls /kaggle/working/
!mkdir /kaggle/working/train/cat

!mv /kaggle/working/train/cat.*.* /kaggle/working/train/cat

!mkdir /kaggle/working/train/dog

!mv /kaggle/working/train/dog.*.* /kaggle/working/train/dog
train_cats_dir = '/kaggle/working/train/cat'

train_dogs_dir = '/kaggle/working/train/dog'
train_cat_fnames = os.listdir( train_cats_dir )

train_dog_fnames = os.listdir( train_dogs_dir )
%matplotlib inline



import matplotlib.image as mpimg

import matplotlib.pyplot as plt



# Parameters for our graph; we'll output images in a 4x4 configuration

nrows = 4

ncols = 4



pic_index = 0 # Index for iterating over images
# Set up matplotlib fig, and size it to fit 4x4 pics

fig = plt.gcf()

fig.set_size_inches(ncols*4, nrows*4)



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

    sp.axis('Off') # Don't show axes (or gridlines)



    img = mpimg.imread(img_path)

    plt.imshow(img)



plt.show()

model = keras.Sequential()



model.add(keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(150, 150, 3)))

model.add(keras.layers.MaxPooling2D(2, 2))

model.add(keras.layers.Conv2D(32, (3, 3), activation='relu'))

model.add(keras.layers.MaxPooling2D(2, 2))

model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))

model.add(keras.layers.MaxPooling2D(2, 2))

model.add(keras.layers.Flatten())

model.add(keras.layers.Dense(512, activation='relu'))

model.add(keras.layers.Dense(1, activation='sigmoid'))
model.compile(optimizer=keras.optimizers.RMSprop(learning_rate=0.001),

              loss='binary_crossentropy',

              metrics=['accuracy'])
from keras.preprocessing.image import ImageDataGenerator



training_datagen = ImageDataGenerator(rescale=1/255)

test_datagen = ImageDataGenerator(rescale=1/255)
train_gen = training_datagen.flow_from_directory('/kaggle/working/train',

                                                 batch_size=20,

                                                 class_mode='binary',

                                                 target_size=(150, 150))



test_gen = test_datagen.flow_from_directory('/kaggle/working/',

                                            batch_size=1,

                                            classes=['test1'],

                                            class_mode=None,

                                            shuffle=False,

                                            target_size=(150, 150))
history = model.fit(train_gen,

                    steps_per_epoch=100,

                    epochs=15,

                    verbose=1)
preds = model.predict(test_gen, batch_size=None, verbose=1, workers=-1)
len(preds)
def get_label(f):

    return f.split('/')[-1].strip('.jpg')
preds
predictions = preds.reshape(12500)
predictions
import pandas as pd

submission = pd.DataFrame({

    'id': [get_label(f) for f in test_gen.filenames],

    'label': predictions

})
submission
submission.to_csv('/kaggle/working/submission.csv')