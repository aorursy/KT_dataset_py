# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

os.chdir('../input')







# Any results you write to the current directory are saved as output.
print(os.getcwd())

print(os.listdir())
print(os.listdir('fer2013-images/images/images_fer2013/Training'))
print(os.listdir('fer2013-images/images/images_fer2013'))
print(os.listdir('xception/'))
train_dir = 'fer2013-images/images/images_fer2013/Training/'

validation_dir = 'fer2013-images/images/images_fer2013/PublicTest/'
from keras.preprocessing.image import ImageDataGenerator



datagen = ImageDataGenerator(

                            rotation_range=40,

                            width_shift_range=0.2,

                            height_shift_range=0.2,

                            shear_range=0.2,

                            zoom_range=0.2,

                            horizontal_flip=True,

                            fill_mode='nearest')
import matplotlib.pyplot as plt

from keras.preprocessing import image



fnames = [os.path.join(train_dir, 'Sad', fname) for fname in os.listdir(os.path.join(train_dir, 'Sad'))]



img_path = fnames[1]



img = image.load_img(img_path, target_size = (48, 48))



x = image.img_to_array(img)



x = x.reshape((1,) + x.shape)



i = 0

for batch in datagen.flow(x, batch_size = 1):

    plt.figure()

    imgplot = plt.imshow(image.array_to_img(batch[0]))

    i += 1

    if i % 4 == 0:

        break

plt.show()
"""from keras.applications.inception_v3 import InceptionV3



conv_base = InceptionV3(weights='inceptionv3/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5',

                       include_top=False,

                       input_shape=(75, 75, 3))

"""
from keras.applications.xception import Xception



conv_base = Xception(weights='xception/xception_weights_tf_dim_ordering_tf_kernels_notop.h5',

                     include_top=False,

                     input_shape=(75, 75, 3))
train_datagen = ImageDataGenerator(

                rescale = 1./255,

                rotation_range=40,

                width_shift_range=0.2,

                height_shift_range=0.2,

                shear_range=0.2,

                zoom_range=0.2,

                horizontal_flip=True)



test_datagen = ImageDataGenerator(rescale=1./255)



train_generator = train_datagen.flow_from_directory(

                                train_dir,

                                target_size = (75, 75),

                                batch_size = 32,

                                class_mode='categorical')



validation_generator = test_datagen.flow_from_directory(

                                validation_dir,

                                target_size = (75, 75),

                                batch_size = 32,

                                class_mode = 'categorical')
conv_base.summary()
from  keras import models

from keras import layers



model = models.Sequential()

model.add(conv_base)

model.add(layers.Flatten())

model.add(layers.Dense(256, activation = 'relu'))

model.add(layers.Dense(7, activation = 'softmax'))
model.summary()
print('This is the number of trainable weights before freezing the conv base: ',len(model.trainable_weights))

conv_base.trainable = False

print('This is the number of trainable weights after freezing the conv base:', len(model.trainable_weights))
from keras.preprocessing.image import ImageDataGenerator

from keras import optimizers



train_datagen = ImageDataGenerator(

                                rescale = 1./255,

                                rotation_range = 40,

                                width_shift_range = 0.2,

                                height_shift_range = 0.2,

                                shear_range = 0.2,

                                zoom_range=0.2,

                                horizontal_flip=True,

                                fill_mode='nearest')



test_datagen = ImageDataGenerator(rescale=1./255)



train_generator = train_datagen.flow_from_directory(

        train_dir,

        target_size=(75, 75),

        color_mode = 'rgb',

        batch_size = 20,

        class_mode = 'categorical')



validation_generator = test_datagen.flow_from_directory(

        validation_dir,

        target_size = (75, 75),

        color_mode = 'rgb',

        batch_size = 20,

        class_mode = 'categorical')



model.compile(loss = 'categorical_crossentropy',

            optimizer = optimizers.RMSprop(lr=2e-5),

             metrics=['acc'])



history = model.fit_generator(

        train_generator,

        steps_per_epoch=1436,

        epochs=30,

        validation_data=validation_generator,

        validation_steps = 180)
conv_base.trainable = True



set_trainable = False

for layer in conv_base.layers:

    if layer.name == 'block14_sepconv1':

        set_trainable = True

    if set_trainable:

        layer.trainable = True

    else:

        layer.trainable = False
model.summary()
model.compile(loss = 'categorical_crossentropy',

             optimizer=optimizers.RMSprop(lr=1e-5),

             metrics=['acc'])



history = model.fit_generator(

    train_generator,

    steps_per_epoch=1436,

    epochs=100,

    validation_data = validation_generator,

    validation_steps=180)
import matplotlib.pyplot as plt



acc = history.history['acc']

val_acc = history.history['val_acc']

loss = history.history['loss']

val_loss = history.history['val_loss']



epochs = range(1, len(acc) + 1)



plt.plot(epochs, acc, 'bo', label='Training acc')

plt.plot(epochs, val_acc, 'b', label='Validation acc')

plt.title('Training and Validation Accuracy')

plt.legend()



plt.figure()



plt.plot(epochs, loss, 'bo', label='Training loss')

plt.plot(epochs, val_loss, 'b', label='Validation loss')

plt.title('Training and Validation loss')

plt.legend()



plt.show()
from keras import layers

from keras import models

from keras import optimizers



model = models.Sequential()

model.add(layers.Conv2D(64, (3, 3), activation = 'relu', input_shape = (48, 48, 3)))

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(64, (3, 3), activation = 'relu'))

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(128, (3, 3), activation = 'relu'))

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(256, (3, 3), activation = 'relu'))

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Flatten())

model.add(layers.Dropout(0.5))

model.add(layers.Dense(512, activation = 'relu'))

model.add(layers.Dense(7, activation = 'softmax'))



model.compile(loss = 'categorical_crossentropy',

             optimizer=optimizers.RMSprop(lr=1e-4),

             metrics=['acc'])
model.summary()
img_path = 'images/images_fer2013/Training/Angry/10002.jpg'



from keras.preprocessing import image

import numpy as np



img = image.load_img(img_path, target_size=(150, 150))

img_tensor = image.img_to_array(img)

img_tensor = np.expand_dims(img_tensor, axis = 0)

img_tensor /= 255.
img_tensor.shape
import matplotlib.pyplot as plt



plt.imshow(img_tensor[0])

plt.show()
nb_train_samples = 28709

nb_validation_samples = 3589

epochs = 50

batch_size = 32



from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(

                rescale = 1./255,

                rotation_range=40,

                width_shift_range=0.4,

                height_shift_range=0.4,

                shear_range=0.2,

                zoom_range=0.2,

                horizontal_flip=True)



test_datagen = ImageDataGenerator(rescale=1./255)



train_generator = train_datagen.flow_from_directory(

                                train_dir,

                                target_size = (48, 48),

                                batch_size = 32,

                                class_mode='categorical')



validation_generator = test_datagen.flow_from_directory(

                                validation_dir,

                                target_size = (48, 48),

                                batch_size = 32,

                                class_mode = 'categorical')



history = model.fit_generator(

                            train_generator,

                            steps_per_epoch = nb_train_samples // batch_size,

                            epochs = 100,

                            validation_data = validation_generator,

                            validation_steps = nb_validation_samples // batch_size)
print(os.getcwd())
os.chdir('../input/fer2013/fer2013')
from datetime import datetime



mod_time = os.stat('fer2013.csv').st_mtime

print(datetime.fromtimestamp(mod_time))
df = pd.read_csv('fer2013.csv')

df.head()
df.dtypes
set(df['Usage'])
train = df[df['Usage'] == 'Training'].copy()
train.head()
train.groupby('emotion').count()['pixels']
from keras.preprocessing import image



train['pixels'] = train['pixels'].apply(lambda m : np.asarray(m.split(' '), dtype = 'float32'))
np.expand_dims(train['pixels'].iloc[0].reshape((48, -1)), axis = 0).shape
expressions = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}
import matplotlib.pyplot as plt

plt.imshow(train['pixels'].iloc[7].reshape((48, 48)) , cmap = 'viridis')

plt.show()
type(image.array_to_img(train['pixels'].iloc[7].reshape((48, 48, 1))))
train['expression'] = train['emotion'].apply(lambda m: expressions[m])
import matplotlib.pyplot as plt

img1 = image.array_to_img(train['pixels'].iloc[7].reshape((48, 48, 1)))

plt.imshow(img1)

#img1.save('{}.7.jpg'.format(train.loc[7, 'expression']))

plt.show()
train['pixels'].iloc[7].reshape((48, 48, 1)).dtype
train.head()
x = '{}.7.jpg'.format(train.loc[7, 'expression'])

type(x)
i = 0

train_images = np.zeros((len(train), 48, 48))

for pixels in train['pixels']:

    train_images[i] = pixels.reshape(48, 48)

    i += 1
train_labels = train['emotion'].values
from keras import layers

from keras import models



model = models.Sequential()

model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)))

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(64, (3, 3), activation = 'relu'))

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(64, (3, 3), activation = 'relu'))
model.summary()
model.add(layers.Flatten())

model.add(layers.Dense(64, activation = 'relu'))

model.add(layers.Dense(7, activation = 'softmax'))



model.summary()
test = df[df['Usage'] == 'PublicTest'].copy()
test['pixels'] = test['pixels'].map(lambda m: np.asarray(m.split(' ')))
i = 0

test_images = np.zeros((len(test), 48, 48))

for pixels in test['pixels']:

    test_images[i] = pixels.reshape((48, 48))

    i+=1

test_images.shape
test_labels = test['emotion'].values
train_images.shape
test_images.shape
train_images = train_images.reshape((28709, 48, 48, 1))

train_images = train_images.astype('float32')/255
test_images = test_images.reshape((3589, 48, 48, 1))

test_images = test_images.astype('float32')/255
from keras.utils import to_categorical



train_labels = to_categorical(train_labels)

test_labels = to_categorical(test_labels)
test_labels.shape
train_labels.shape
model.compile(optimizer='rmsprop',

             loss = 'categorical_crossentropy',

             metrics = ['accuracy'])

model.fit(train_images, train_labels, epochs = 5, batch_size = 32)
test_loss, test_acc = model.evaluate(test_images, test_labels)
import numpy as np

np.bincount(df['emotion'])



### Disgust has a very low count!
print(test_acc)