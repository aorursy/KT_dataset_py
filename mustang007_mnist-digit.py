# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

# import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv('/kaggle/input/mnist-digit-dataset/MNIST_DIGIT_DATSET/train.csv')
data.head()
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import os
from tensorflow.keras import layers
from tensorflow.keras import Model
import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3

from tensorflow.keras.optimizers import RMSprop


from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img

from sklearn.model_selection import train_test_split

from keras.preprocessing.image import ImageDataGenerator
sns.countplot(data['label'])

weight_file = '/kaggle/input/inception-v3/inception_v3.ckpt'
pre_trained_model = InceptionV3(input_shape = (75,75,3),
                                include_top = False,
                               weights = 'imagenet')

# pre_trained_model.load_weights(weight_file)
for layer in pre_trained_model.layers:
      layer.trainable = False
last_layer = pre_trained_model.get_layer('mixed7')
last_output = last_layer.output

x = layers.Flatten()(last_output)

x = layers.Dense(1024, activation='relu')(x)

x = layers.Dropout(0.2)(x)

x = layers.Dense(10, activation='softmax')(x)

model = Model(pre_trained_model.input, x)

# model.summary()

# model = tf.keras.models.Sequential([
#     # Note the input shape is the desired size of the image 150x150 with 3 bytes color
#     # This is the first convolution
#     tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28, 1)),
#     tf.keras.layers.MaxPooling2D(2, 2),
#     # The second convolution
#     tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
#     tf.keras.layers.MaxPooling2D(2,2),
#     # The third convolution
#     tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
#     tf.keras.layers.MaxPooling2D(2,2),
   
#     # Flatten the results to feed into a DNN
#     tf.keras.layers.Flatten(),
#     tf.keras.layers.Dropout(0.5),
#     # 512 neuron hidden layer
#     tf.keras.layers.Dense(512, activation='relu'),
#     tf.keras.layers.Dense(10, activation='softmax')
# ])

model.compile(optimizer = RMSprop(lr = 0.0001),
             loss = 'sparse_categorical_crossentropy',
             metrics = ['accuracy'])


train_folder =  '/kaggle/input/mnist-digit-dataset/MNIST_DIGIT_DATSET/Images/train/'
test_folder =  '/kaggle/input/mnist-digit-dataset/MNIST_DIGIT_DATSET/Images/test/'
data['image_path'] = data.apply(lambda x: (train_folder + x['filename']), axis=1)

train_data = np.array([img_to_array(load_img(img, target_size=(28,28)))
                      for img in data['image_path'].values.tolist()]).astype('float32')
train_label = data['label']
# Split the data into train and validation. The stratify parm will insure  train and validation  
# will have the same proportions of class labels as the input dataset.
x_train, x_validation, y_train, y_validation = train_test_split(train_data, train_label, test_size=0.2, stratify=np.array(train_label), random_state=100)
print('x_train shape = ',x_train.shape)
print('y_train shape = ',y_train.shape)
print('x_validation shape = ',x_validation.shape)
print('y_validation shape = ',y_validation.shape)
train_datagen = ImageDataGenerator(rescale=1./255,
                                  rotation_range=30,
                                  width_shift_range = 0.2,
                                  height_shift_range = 0.2)

val_datagen = ImageDataGenerator(rescale = 1./255)



train_gen = train_datagen.flow(x_train,
                               y_train,
#                                class_mode='categorical',
                              batch_size=126)


val_gen = val_datagen.flow(x_validation,
                           y_validation, 
#                            class_mode='categorical',
                           batch_size=126)


history = model.fit_generator(train_gen,
                              epochs =15,
                   steps_per_epoch=200,
                   validation_data = val_gen,
                    validation_steps = 44,
                   
                   verbose = 2)