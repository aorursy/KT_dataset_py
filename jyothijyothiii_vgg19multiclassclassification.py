# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

#for dirname, _, filenames in os.walk('/kaggle/input'):

#    for filename in filenames:

#        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np

import pandas as pd

import tensorflow as tf

import keras 

from keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.layers import Input



tf.__version__
# Part 1 - Data Preprocessing



# Preprocessing the Training set

train_datagen = ImageDataGenerator(rescale = 1./255,

                                   shear_range = 0.2,

                                   zoom_range = 0.2,

                                   horizontal_flip = True)

training_set = train_datagen.flow_from_directory('../input/intel-image-classification/seg_train/seg_train',

                                                 target_size = (224,224),

                                                 batch_size = 32,

                                                 shuffle = True,

                                                 class_mode = 'categorical')



# Preprocessing the Test set

test_datagen = ImageDataGenerator(rescale = 1./255)

test_set = test_datagen.flow_from_directory('../input/intel-image-classification/seg_test/seg_test',

                                            target_size = (224, 224)

                                            )

# Part 2 - Building the CNN

input_tensor = Input(shape=(224, 224, 3))

vgg_weights = "../input/vgg19/vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5"

base_model = tf.keras.applications.VGG19(

    input_tensor=input_tensor,

    include_top  = False,

    weights      = None,

)

base_model.load_weights(vgg_weights)

x = base_model.output

print(base_model.input)

print(base_model.output)
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D

#base_model.trainable = False

x = tf.keras.layers.GlobalAveragePooling2D()

#x = Dense(1024, activation='relu')(x)

predictions = tf.keras.layers.Dense(6,activation="softmax")
from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Flatten,Dense

model = Sequential([base_model,

                   x,

                    predictions

                ])

print(model)

model.summary()
base_model.summary()
#base_model.trainable = False

for layer in base_model.layers:

    layer.trainable = False
base_model.summary()
def get_callbacks(name):

    return[

        tfdocs.modeling.EpochDots(),

        tf.keras.callbacks.EarlyStopping(monitor='val_categorical_crossentropy',patience=1),

    ]
base_learning_rate  = 0

model.compile(optimizer=tf.keras.optimizers.Adam(), loss="categorical_crossentropy",metrics=[tf.keras.losses.CategoricalCrossentropy(name='categorical_crossentropy'),'accuracy'])

history = model.fit(x = training_set,epochs= 8, validation_data = test_set)
classes =["buildings","forest","glacier","mountain","sea","street"]
model.evaluate(test_set)
print(training_set.class_indices)

#cnn.save("mobilenetmodel1.h5")
import matplotlib.pyplot as plt

# Accuracy/validation plots

h = history.history

fig = plt.figure(figsize = (13, 4))



plt.subplot(121)

plt.plot(h['accuracy'], label = 'acc')

plt.plot(h['val_accuracy'], label = 'val_acc')

plt.legend()

plt.grid(False)

plt.title(f'Training and validation accuracy')



plt.subplot(122)

plt.plot(h['loss'], label = 'loss')

plt.plot(h['val_loss'], label = 'val_loss')

plt.legend()

plt.grid(False)

plt.title(f'Training and Test loss')
import os

pred_dir = "../input/intel-image-classification/seg_pred/seg_pred/"

import random

pred_files = random.sample(os.listdir(pred_dir),5)

from keras.preprocessing import image

i =1

for f in pred_files:

    path = str(pred_dir+f)

    test_image = image.load_img(path, target_size = (224, 224))

    test_image = image.img_to_array(test_image)

    test_image = np.expand_dims(test_image, axis = 0)

    result = model.predict(test_image)

    img = image.load_img(path)

    num_rows = 5

    num_cols = 3

    num_images = num_rows,num_cols

    plt.figure(figsize = (6,3))

    plt.subplot(1,1,i)

    plt.grid(False)

    plt.imshow(img)

    plt.show()

    classes =["buildings","forest","glacier","mountain","sea","street"]    

    print("The predicted image is :" + str(classes[np.argmax(np.array(result))]))

    

    