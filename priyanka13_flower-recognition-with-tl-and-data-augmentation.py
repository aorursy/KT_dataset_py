# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import tensorflow

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Flatten, Conv2D

from sklearn.model_selection import train_test_split

from tensorflow.keras.applications.resnet50 import ResNet50,preprocess_input

from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

import os

import pandas as pd

import shutil
%mkdir -p data/train

%mkdir -p data/valid

%cd data/

%mkdir -p train/daisy

%mkdir -p train/dandelion

%mkdir -p train/rose

%mkdir -p train/sunflower

%mkdir -p train/tulip



%mkdir -p valid/daisy

%mkdir -p valid/dandelion

%mkdir -p valid/rose

%mkdir -p valid/sunflower

%mkdir -p valid/tulip
class Preprocessing:

    

    def __init__(self,base_path,categories,t_path,v_path):

        self.categories = categories

        self.path = base_path

        self.train_path = t_path

        self.validation_path = v_path

        self.df_list = []

        

    def load_flowers_data(self):   

        for flower in self.categories:

            images = []  

            images = os.listdir(self.path+"/"+flower)

            for image in images:

                self.df_list.append((flower,self.path+"/"+flower+"/"+image))

        flowers = pd.DataFrame(self.df_list,columns=['category','image_path'])

        return flowers

    

    def copy_util(self,df,category,dest):

        for idx,row in df.iterrows():

            shutil.copy(row["image_path"],dest+"/"+category+"/"+row["image_path"].split("/")[-1])

            

    

    def random_image_split(self,flowers):

        for category in categories:

            valid_data = flowers_data[flowers_data["category"]==category].sample(frac=0.2).reset_index(drop=True)

            self.copy_util(valid_data,category,self.validation_path)

            val_idx = valid_data.index

            train_data = flowers_data[(~flowers_data.index.isin(val_idx)) & (flowers_data["category"] == category)]

            self.copy_util(train_data,category,self.train_path)            

            

        
base_path = "/kaggle/input/flowers-recognition/flowers/flowers"

train_path = "/kaggle/working/data/train"

validation_path = "/kaggle/working/data/valid"

categories = os.listdir(base_path)

prep = Preprocessing(base_path,categories,train_path,validation_path)

flowers_data = prep.load_flowers_data()

prep.random_image_split(flowers_data)
num_classes = 5

resnet_weights_path = '/kaggle/input/resnet50/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'



flowers_model = Sequential()

flowers_model.add(ResNet50(include_top=False, pooling='avg', weights=resnet_weights_path))

flowers_model.add(Dense(num_classes, activation='softmax'))



flowers_model.compile(optimizer='sgd', 

                     loss='categorical_crossentropy', 

                     metrics=['accuracy'])





flowers_model.layers[0].trainable = False
image_size = 224

data_generator = ImageDataGenerator(preprocessing_function=preprocess_input,

                                              horizontal_flip = True)

train_generator = data_generator.flow_from_directory(

                                        directory='/kaggle/working/data/train',

                                        target_size=(image_size, image_size),

                                        batch_size=100,

                                        class_mode='categorical')



validation_generator = data_generator.flow_from_directory(

                                        directory='/kaggle/working/data/valid',

                                        target_size=(image_size, image_size),

                                        class_mode='categorical')

fit_stats = flowers_model.fit_generator(train_generator,

                                       steps_per_epoch=41,

                                       validation_data=validation_generator,

                                       validation_steps=1)