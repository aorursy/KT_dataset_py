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
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.layers import Dense,Flatten,Conv2D,MaxPool2D

from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.applications.vgg19 import VGG19

from tensorflow.keras.callbacks import Callback
class MyCallbacks(Callback):

    def on_epoch_end(self, epoch, logs={}):

        if logs.get('accuracy') > 0.95:

            self.model.stop_training = False

        

callbacks = MyCallbacks()
pre_trained = VGG19(include_top=False, input_shape=(150,150,3))



for layer in pre_trained.layers:

    layer.trainable = False

print('Done')
l = pre_trained.output



l = Flatten()(l)

l = Dense(512, 'relu')(l)

l = Dense(512, 'relu')(l)

l = Dense(512, 'relu')(l)



pred = Dense(6, 'softmax')(l)
model = Model(pre_trained.input, pred)
model.compile(optimizer='adam',

             loss='categorical_crossentropy',

             metrics=['accuracy'])
model.summary()
train_dir = '../input/intel-image-classification/seg_train/seg_train'

test_dir = '../input/intel-image-classification/seg_test/seg_test'
from tensorflow.keras.preprocessing.image import ImageDataGenerator,load_img,img_to_array
train_gen = ImageDataGenerator(rescale=1./255,

                              width_shift_range=0.3,

                              height_shift_range=0.3,

                              zoom_range=0.3,

                              shear_range=0.2,

                              horizontal_flip=True)



test_gen = ImageDataGenerator(rescale=1./255)
batch = 32
train_set = train_gen.flow_from_directory(train_dir,

                                            batch_size=batch,

                                            target_size=(150,150))



test_set = test_gen.flow_from_directory(test_dir,

                                       batch_size=batch,

                                       target_size=(150,150))
model.fit(train_set,

         validation_data = test_set,

         steps_per_epoch=14034/batch,

         epochs=20,

         verbose=1,

         callbacks=[callbacks])
def predict(model, path):

    

    img = load_img(path,target_size=(150,150,3))

    arr_img = img_to_array(img)

    arr_img = np.expand_dims(arr_img, axis=0)

    pred = model.predict(arr_img)

    plt.imshow(img)

    

    if pred[0][0] > 0.5:

        print('Buildings')

    elif pred[0][1] > 0.5:

        print('Forest')

    elif pred[0][2] > 0.5:

        print('Glacier')

    elif pred[0][3] > 0.5:

        print('Mountain')

    elif pred[0][4] > 0.5:

        print('Sea')

    elif pred[0][5] > 0.5:

        print('Street')
predict(model, '../input/intel-image-classification/seg_pred/seg_pred/10021.jpg')
predict(model, '../input/intel-image-classification/seg_pred/seg_pred/10047.jpg')
predict(model, '../input/intel-image-classification/seg_pred/seg_pred/10073.jpg')