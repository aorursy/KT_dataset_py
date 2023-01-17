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
path1 = "/kaggle/input/mura-final/finaldata/train_final/XR_FINGER"

path2 = "/kaggle/input/mura-final/finaldata/valid_final/XR_FINGER"
import tensorflow as tf

print(tf.__version__)
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.applications.imagenet_utils import preprocess_input



def preprocess(image):

  k = preprocess_input(image,mode="torch")

  return k



im_generator = tf.keras.preprocessing.image.ImageDataGenerator(horizontal_flip=True,vertical_flip=True,rotation_range=40,preprocessing_function=preprocess)
im_generator_ = tf.keras.preprocessing.image.ImageDataGenerator(preprocessing_function=preprocess)
train_generator = im_generator.flow_from_directory(path1,

                                                    batch_size=8,

                                                    shuffle=True,

                                                    color_mode="rgb",

                                                    class_mode = 'binary',

                                                    target_size=(300,300)

                                                    

)
valid_generator = im_generator_.flow_from_directory(path2,

                                                    batch_size=8,

                                                    shuffle=True,

                                                    color_mode="rgb",

                                                    class_mode = 'binary',

                                                    target_size=(300,300)

)
! pip install -U git+https://github.com/qubvel/efficientnet



import efficientnet.tfkeras as efn 

model = efn.EfficientNetB7(weights='noisy-student',include_top = False,input_shape = (300,300,3))



model_final = tf.keras.Sequential([

                                   model,

                                   tf.keras.layers.AveragePooling2D(10),

                                   tf.keras.layers.Flatten(),

                                   tf.keras.layers.Dropout(0.5),

                                   tf.keras.layers.Dense(1,activation=tf.nn.sigmoid)

])
model_final.summary()

import tensorflow_addons as tfa
model_final.compile(optimizer=tf.keras.optimizers.Adam(),loss = tf.keras.losses.BinaryCrossentropy(),metrics = ['accuracy','mse',tfa.metrics.CohenKappa(num_classes=2)])
! mkdir models
c1 = tf.keras.callbacks.ModelCheckpoint(

    filepath="/kaggle/working/models/weights.{epoch:02d}.h5", 

    verbose=1, 

    save_weights_only=False)
history = model_final.fit(train_generator,epochs=1,steps_per_epoch=int(5106//8),validation_data=valid_generator,validation_steps=461//8,callbacks = [c1])
df1 = pd.DataFrame.from_dict(history.history)

df1.to_csv("history.csv")