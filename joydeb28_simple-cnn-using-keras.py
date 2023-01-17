# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
import zipfile
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import TensorBoard,EarlyStopping
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
class LoadData():
    def __init__(self):
        self.train_file = None
        self.test_file = None
        self.submisson_File = None
        self.submisson_File = os.path.join('/kaggle/input/dogs-vs-cats',"Submission.csv")
        self.data_frame = None
        self.train_df = None
        self.validation_df = None
        
        zip_ref = zipfile.ZipFile('/kaggle/input/dogs-vs-cats/train.zip', 'r')
        zip_ref.extractall('/tmp/train')
        zip_ref.close()
        
    def prepare_data(self):
        filenames = os.listdir("/tmp/train/train")
        categories = []
        for filename in filenames:
            category = filename.split('.')[0]
            if category == 'dog':
                categories.append(1)
            else:
                categories.append(0)

        self.data_frame = pd.DataFrame({'filename': filenames,'category': categories})
        self.data_frame["category"] = self.data_frame["category"].replace({0: 'cat', 1: 'dog'}) 
        self.train_df, self.validation_df = train_test_split(self.data_frame, test_size=0.20, random_state=42)
        self.train_df = self.train_df.reset_index(drop=True)
        self.validation_df = self.validation_df.reset_index(drop=True)

ld_obj = LoadData()
ld_obj.prepare_data()
ld_obj.data_frame.head()
ld_obj.train_df.head()
ld_obj.validation_df.head()
class PreProcessing():
    def __init__(self):
        self.train_data_gen = None
        self.valid_data_gen = None
        self.FAST_RUN = False
        self.image_width=128
        self.image_height=128
        self.image_size=(self.image_width, self.image_height)
        self.image_channels=3
        self.batch_size = 15
        self.train_data_length = ld_obj.train_df.shape[0]
        self.validation_data_length = ld_obj.validation_df.shape[0]
        self.train_generator = None
        self.validation_generator = None
        
    def data_genrator(self):
        train_datagen = ImageDataGenerator(
        rotation_range=15,
        rescale=1./255,
        shear_range=0.1,
        zoom_range=0.2,
        horizontal_flip=True,
        width_shift_range=0.1,
        height_shift_range=0.1
        )

        self.train_generator = train_datagen.flow_from_dataframe(
            ld_obj.train_df, 
            "/tmp/train/train/", 
            x_col='filename',
            y_col='category',
            target_size=self.image_size,
            class_mode='categorical',
            batch_size=self.batch_size
        )
        
        validation_datagen = ImageDataGenerator(rescale=1./255)
        self.validation_generator = validation_datagen.flow_from_dataframe(
            ld_obj.validation_df, 
            "/tmp/train/train/", 
            x_col='filename',
            y_col='category',
            target_size=self.image_size,
            class_mode='categorical',
            batch_size=self.batch_size
        )

pre_process_obj = PreProcessing()
pre_process_obj.data_genrator()
class DesignModel():
    def __init__(self):
        self.model = None
        
    def create_model(self):
        model = Sequential()

        model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(pre_process_obj.image_width, pre_process_obj.image_height, pre_process_obj.image_channels)))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(128, (3, 3), activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        
        model.add(Dense(2, activation='softmax'))
        self.model = model
    
    def compile_model(self):
        self.model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

        self.model.summary()
    def train_model(self):
        
        self.model.fit(pre_process_obj.train_generator,
                              validation_data=pre_process_obj.validation_generator,
                              steps_per_epoch=pre_process_obj.train_data_length//pre_process_obj.batch_size,
                              epochs=15,
                              validation_steps=pre_process_obj.validation_data_length//pre_process_obj.batch_size,
                              verbose=2)
    
model_obj = DesignModel()
model_obj.create_model()
model_obj.compile_model()
model_obj.train_model()
class Prediction():
    def __init__(self,model):
        self.model = model
        
    def prediction(self):
        zip_ref = zipfile.ZipFile('/kaggle/input/dogs-vs-cats/test1.zip', 'r')
        zip_ref.extractall('/tmp/test1')
        zip_ref.close()
        test_filenames = os.listdir("/tmp/test1/test1")
        test_df = pd.DataFrame({
            'filename': test_filenames
        })
        nb_samples = test_df.shape[0]
        
        test_gen = ImageDataGenerator(rescale=1./255)
        test_generator = test_gen.flow_from_dataframe(
            test_df, 
            "/tmp/test1/test1/", 
            x_col='filename',
            y_col=None,
            class_mode=None,
            target_size=pre_process_obj.image_size,
            batch_size=pre_process_obj.batch_size,
            shuffle=False
        )
        predict = self.model.predict_generator(test_generator, steps=np.ceil(nb_samples/pre_process_obj.batch_size))
        test_df['category'] = np.argmax(predict, axis=-1)
        label_map = dict((v,k) for k,v in pre_process_obj.train_generator.class_indices.items())
        test_df['category'] = test_df['category'].replace(label_map)
        test_df['category'] = test_df['category'].replace({ 'dog': 1, 'cat': 0 })

        submission_df = test_df.copy()
        submission_df['id'] = submission_df['filename'].str.split('.').str[0]
        submission_df['label'] = submission_df['category']
        submission_df.drop(['filename', 'category'], axis=1, inplace=True)
        submission_df.to_csv('submission.csv', index=False)
        
    def predict(self,image):
        print(image)
        
pred_obj = Prediction(model_obj.model)
pred_obj.prediction()