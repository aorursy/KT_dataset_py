# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train=pd.read_csv("../input/aptos2019-blindness-detection/train.csv")

test=pd.read_csv("../input/aptos2019-blindness-detection/test.csv")

submission_df=pd.read_csv("../input/aptos2019-blindness-detection/sample_submission.csv")
import matplotlib.pyplot as plt

import seaborn as sns

sns.countplot(x='diagnosis',data=train)
!mkdir ../data

!mkdir ../data/train



!mkdir ../data/train/0

!mkdir ../data/train/1

!mkdir ../data/train/2

!mkdir ../data/train/3

!mkdir ../data/train/4
import subprocess

from tqdm import tqdm



def move_img(df,kind):

    for id_code ,diagnosis in tqdm(zip(df['id_code'],df['diagnosis'])):

        if diagnosis == 0:

            subprocess.call(['cp','../input/aptos2019-blindness-detection/{}_images/{}.png'.format(kind,id_code),'../data/{}/0/{}.png'.format(kind,id_code)])

        if diagnosis == 1:

            subprocess.call(['cp','../input/aptos2019-blindness-detection/{}_images/{}.png'.format(kind,id_code),'../data/{}/1/{}.png'.format(kind,id_code)])

        if diagnosis == 2:

            subprocess.call(['cp','../input/aptos2019-blindness-detection/{}_images/{}.png'.format(kind,id_code),'../data/{}/2/{}.png'.format(kind,id_code)])

        if diagnosis == 3:

            subprocess.call(['cp','../input/aptos2019-blindness-detection/{}_images/{}.png'.format(kind,id_code),'../data/{}/3/{}.png'.format(kind,id_code)])

        if diagnosis == 4:

            subprocess.call(['cp','../input/aptos2019-blindness-detection/{}_images/{}.png'.format(kind,id_code),'../data/{}/4/{}.png'.format(kind,id_code)])
move_img(train,'train')
import tensorflow as tf

print("tensorflow version: {}".format(tf.__version__))



from tensorflow import keras

print("keras version: {}".format(keras.__version__))



import numpy as np

print("numpy version: {}".format(np.__version__))



from keras import layers

from keras import models

from keras import optimizers

from keras.preprocessing.image import ImageDataGenerator



import matplotlib.pyplot as plt

%matplotlib inline
import numpy as np





class MixupImageDataGenerator():

    def __init__(self, generator, directory, batch_size, img_height, img_width, alpha=0.2, subset=None):

        """Constructor for mixup image data generator.



        Arguments:

            generator {object} -- An instance of Keras ImageDataGenerator.

            directory {str} -- Image directory.

            batch_size {int} -- Batch size.

            img_height {int} -- Image height in pixels.

            img_width {int} -- Image width in pixels.



        Keyword Arguments:

            alpha {float} -- Mixup beta distribution alpha parameter. (default: {0.2})

            subset {str} -- 'training' or 'validation' if validation_split is specified in

            `generator` (ImageDataGenerator).(default: {None})

        """



        self.batch_index = 0

        self.batch_size = batch_size

        self.alpha = alpha



        # First iterator yielding tuples of (x, y)

        self.generator1 = generator.flow_from_directory(directory,

                                                        target_size=(

                                                            img_height, img_width),

                                                        class_mode="categorical",

                                                        batch_size=batch_size,

                                                        shuffle=True,

                                                        subset=subset)



        # Second iterator yielding tuples of (x, y)

        self.generator2 = generator.flow_from_directory(directory,

                                                        target_size=(

                                                            img_height, img_width),

                                                        class_mode="categorical",

                                                        batch_size=batch_size,

                                                        shuffle=True,

                                                        subset=subset)



        # Number of images across all classes in image directory.

        self.n = self.generator1.samples



    def reset_index(self):

        """Reset the generator indexes array.

        """



        self.generator1._set_index_array()

        self.generator2._set_index_array()



    def on_epoch_end(self):

        self.reset_index()



    def reset(self):

        self.batch_index = 0



    def __len__(self):

        # round up

        return (self.n + self.batch_size - 1) // self.batch_size



    def get_steps_per_epoch(self):

        """Get number of steps per epoch based on batch size and

        number of images.



        Returns:

            int -- steps per epoch.

        """



        return self.n // self.batch_size



    def __next__(self):

        """Get next batch input/output pair.



        Returns:

            tuple -- batch of input/output pair, (inputs, outputs).

        """



        if self.batch_index == 0:

            self.reset_index()



        current_index = (self.batch_index * self.batch_size) % self.n

        if self.n > current_index + self.batch_size:

            self.batch_index += 1

        else:

            self.batch_index = 0



        # random sample the lambda value from beta distribution.

        l = np.random.beta(self.alpha, self.alpha, self.batch_size)



        X_l = l.reshape(self.batch_size, 1, 1, 1)

        y_l = l.reshape(self.batch_size, 1)



        # Get a pair of inputs and outputs from two iterators.

        X1, y1 = self.generator1.next()

        X2, y2 = self.generator2.next()



        # Perform the mixup.

        X = X1 * X_l + X2 * (1 - X_l)

        y = y1 * y_l + y2 * (1 - y_l)

        return X, y



    def __iter__(self):

        while True:

            yield next(self)
input_generator = ImageDataGenerator(

    preprocessing_function=keras.applications.mobilenet_v2.preprocess_input,

    rotation_range=5,

    width_shift_range=0.2,

    height_shift_range=0.2,

    shear_range=0.05,

    zoom_range=0.2,

    brightness_range=(1, 1.3),

    horizontal_flip=True,

    fill_mode='nearest',

    validation_split=0.15

)



# Note that the validation data should not be augmented!

#test_datagen = ImageDataGenerator(preprocessing_function=keras.applications.vgg16.preprocess_input)



batch_size = 32

img_height = 224

img_width = 224

train_dir = '../data/train'



train_generator = MixupImageDataGenerator(

    generator=input_generator,                         

    directory=train_dir,

    batch_size=batch_size,

    img_height=img_height,

    img_width=img_width,

    subset='training'

)



validation_generator = input_generator.flow_from_directory(

    train_dir,

    target_size=(img_height, img_width),

    class_mode="categorical",

    batch_size=batch_size,

    shuffle=True,    

    subset='validation'

)



print('training steps: ', train_generator.get_steps_per_epoch())

print('validation steps: ', validation_generator.samples // batch_size)
from keras.applications import DenseNet121



def get_pretrained_model():

    conv_base = DenseNet121(

        weights='../input/keras-applications-weights/densenet121_weights_tf_dim_ordering_tf_kernels_notop.h5',

        include_top=False,

        input_shape=(224, 224, 3)

    )

    conv_base.trainable = False

    model = models.Sequential()

    model.add(conv_base)

    model.add(layers.GlobalAveragePooling2D())

#     model.add(layers.Dropout(0.5))

#     model.add(layers.Dense(2048, activation='relu'))

    model.add(layers.Dropout(0.5))

    model.add(layers.Dense(5, activation='softmax'))

    return model
model = get_pretrained_model()

model.compile(loss='categorical_crossentropy',

              optimizer=optimizers.RMSprop(lr=0.001),

              metrics=['acc'])

model.summary()
early_stopping = keras.callbacks.EarlyStopping(

    monitor='val_loss',

    min_delta=0,

    patience=10,

    verbose=0,

    mode='auto',

    baseline=None,

    restore_best_weights=False

)



model_checkpoint = keras.callbacks.ModelCheckpoint(

filepath='model.hdf5',

monitor='val_loss',

verbose=1,

save_best_only=True,

mode='min',

period=1

)



reducelrtop = keras.callbacks.ReduceLROnPlateau(

    monitor='val_loss',

    factor=0.1,

    patience=10,

    verbose=0,

    mode='auto',

    min_delta=0.0001,

    cooldown=0,

    min_lr=0

)
epochs = 2



history = model.fit_generator(

      train_generator,

      steps_per_epoch=train_generator.get_steps_per_epoch(),

      epochs=epochs,

      max_queue_size=10,

      workers=4,

      validation_data=validation_generator,

      validation_steps=validation_generator.samples // batch_size,

      #callbacks=[model_checkpoint, early_stopping],

      use_multiprocessing=True

)
for layer in model.layers:

    layer.trainable = True



model.compile(loss='categorical_crossentropy',

              optimizer=optimizers.RMSprop(lr=0.0001),

              metrics=['acc'])

model.summary()
epochs = 30



history = model.fit_generator(

      train_generator,

      steps_per_epoch=train_generator.get_steps_per_epoch(),

      epochs=epochs,

      max_queue_size=10,

      workers=4,

      validation_data=validation_generator,

      validation_steps=validation_generator.samples // batch_size,

      #callbacks=[model_checkpoint, early_stopping],

      use_multiprocessing=True

)
hist = history



fig, axes = plt.subplots(1, 2, figsize=(20, 5))

ax = axes[0]

ax.plot(hist.history['acc'], 'r', label='Training acc')

ax.plot(hist.history['val_acc'], 'g', label='Validation acc')

ax.set_title('Training and validation accuracy')

ax.legend()



ax = axes[1]

ax.plot(hist.history['loss'], 'r', label='Training loss')

ax.plot(hist.history['val_loss'], 'g', label='Validation loss')

ax.set_title('Training and validation loss')

ax.legend()



plt.show()
test_datagen = ImageDataGenerator(preprocessing_function=keras.applications.mobilenet_v2.preprocess_input)



input_dir = '../input/aptos2019-blindness-detection'

test_generator = test_datagen.flow_from_directory(

    input_dir,

    target_size=(224, 224),

    color_mode="rgb",

    batch_size=1,

    shuffle = False,

    class_mode='categorical',

    classes=['test_images']

)



filenames = test_generator.filenames

nb_samples = len(filenames)

predict = model.predict_generator(test_generator,steps = nb_samples)
submission_df.diagnosis = predict.argmax(1)



submission_df.to_csv('submission.csv',index=False)