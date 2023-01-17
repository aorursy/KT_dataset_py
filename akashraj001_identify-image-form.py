# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load

import os

import pandas as pd

from sklearn.model_selection import train_test_split

from shutil import copyfile

import shutil

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
train_csv=pd.read_csv("/kaggle/input/identify-the-dance-form/train.csv")

train_csv['target'].value_counts()



X=train_csv['Image']

y=train_csv['target']



os.mkdir('/kaggle/final_train_dir')

for i in train_csv['target'].unique():

    os.mkdir('/kaggle/final_train_dir/'+i)

    

for i in train_csv['target'].unique():

    for j in X[y==i]:

        copyfile('/kaggle/input/identify-the-dance-form/train/'+j, '/kaggle/final_train_dir/'+i+'/'+j)
os.listdir('/kaggle/final_train_dir')
import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.applications.vgg16 import VGG16

from tensorflow.keras.layers import Flatten,Dense,Dropout

from tensorflow.keras.models import Model,Sequential

from tensorflow.keras.callbacks import ReduceLROnPlateau

from tensorflow.keras.optimizers import Adam
reduce_learning_rate = ReduceLROnPlateau(monitor='loss',

                                        factor=0.1,

                                        patience=2,

                                        cooldown=2,

                                        min_lr=0.00001,

                                        verbose=1)



callbacks = [reduce_learning_rate]



vggmodel =VGG16(weights='imagenet', include_top=False, input_shape = (224, 224, 3),pooling='max')



vggmodel.trainable = False

model = Sequential([

  vggmodel, 

  Dense(1024, activation='relu'),

  Dropout(0.15),

  Dense(256, activation='relu'),

  Dropout(0.15),

  Dense(8, activation='softmax'),

])



model.compile( optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
train_datagenerator = ImageDataGenerator(

        rescale=1. / 255,

        featurewise_center=False,  

        samplewise_center=False,  

        featurewise_std_normalization=False,  

        samplewise_std_normalization=False,  

        rotation_range=40,  

        zoom_range = 0.20,  

        width_shift_range=0.10,  

        height_shift_range=0.10,  

        horizontal_flip=True,  

        vertical_flip=False) 



image_size=224

train_generator=train_datagenerator.flow_from_directory(

        r"/kaggle/final_train_dir",

        target_size=(image_size,image_size),

#        batch_size=128,

        class_mode='categorical'

        )



history =model.fit_generator(

    train_generator,

    verbose=1,

    epochs=40,

   callbacks=callbacks

)
shutil.copytree('/kaggle/input/identify-the-dance-form/test', '/kaggle/test/images')
test_datagen=ImageDataGenerator(rescale=1/255)

test_generator=test_datagen.flow_from_directory(

        r'/kaggle/test',

        target_size=(image_size,image_size),

#       color_mode="rgb",

        batch_size=32,

        class_mode=None,

        shuffle=False

        )





pred=model.predict_generator(test_generator,verbose=1)

predicted_class_indices=np.argmax(pred,axis=1)



labels = (train_generator.class_indices)



labels = dict((v,k) for k,v in labels.items())

predictions = [labels[k] for k in predicted_class_indices]

filenames=test_generator.filenames

results=pd.DataFrame({"Filename":filenames,

                      "Predictions":predictions})

results["Filename"]=results["Filename"].apply(lambda x:x[7:])



test_csv=pd.read_csv("/kaggle/input/identify-the-dance-form/test.csv")



results.set_index(["Filename"],inplace=True)

test_csv.set_index(["Image"],inplace=True)



output=test_csv.merge(results,left_index=True,right_index=True)

output.index.name='Image'

output.rename(columns={'Predictions':'target'},inplace=True)

output.to_csv('submission.csv')