# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split

#loading training and testing datasets
df_train = pd.read_csv('/kaggle/input/rsna-bone-age/boneage-training-dataset.csv')
df_test = pd.read_csv('/kaggle/input/rsna-bone-age/boneage-test-dataset.csv')

#appending png file extension to id column for both training and testing datasets
df_train['id'] = df_train['id'].apply(lambda x: str(x)+'.png')
df_test['Case ID'] = df_test['Case ID'].apply(lambda x: str(x)+'.png')

#Feature Engineering
df_train['Sex'] = df_train['male'].apply(lambda x: 'M' if x else 'F')
del(df_train['male'])
df_test['id'] = df_test['Case ID']
del(df_test['Case ID'])

#splitting train datasets into traininng and validation datasets
train_df, valid_df = train_test_split(df_train, test_size = 0.2, random_state = 0)

#packages required for image preprocessing
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.resnet50 import preprocess_input
from keras.metrics import mean_absolute_error

image_size = 256

train_data_generator = ImageDataGenerator(preprocessing_function = preprocess_input)
val_data_generator = ImageDataGenerator(preprocessing_function = preprocess_input)
test_data_generator = ImageDataGenerator(preprocessing_function = preprocess_input)

#train data generator
train_generator = train_data_generator.flow_from_dataframe(
    dataframe = train_df,
    directory="../input/rsna-bone-age/boneage-training-dataset/boneage-training-dataset/",
    validate_filenames = False,
    x_col= 'id',
    y_col= 'boneage',
    batch_size = 56,
    flip_vertical = True,
    class_mode = 'other',
    target_size = (image_size, image_size)
)

#validation data generator
val_generator = val_data_generator.flow_from_dataframe(
    dataframe = valid_df,
    directory="../input/rsna-bone-age/boneage-training-dataset/boneage-training-dataset/",
    validate_filenames = False,
    x_col = 'id',
    y_col = 'boneage',
    batch_size = 140,
    flip_vertical = True,
    class_mode = 'other',
    target_size = (image_size, image_size)
)

#test data generator
test_generator = test_data_generator.flow_from_dataframe(
    dataframe = df_test,
    directory="../input/rsna-bone-age/boneage-test-dataset/boneage-test-dataset/",
    validate_filenames = False,
    x_col = 'id',
    y_col = None,
    flip_vertical = True,
    class_mode = None,
    target_size = (image_size, image_size)
)

from tensorflow.python.keras.models import Sequential 
from tensorflow.python.keras.layers import GlobalMaxPooling2D, Dense, Flatten, GlobalAveragePooling2D

#Model definition

my_model = Sequential() 
my_model.add(ResNet50(input_shape=(image_size, image_size, 3), include_top=False, weights='imagenet')) 
my_model.add(GlobalMaxPooling2D())
my_model.add(Flatten()) 
my_model.add(Dense(128, activation='relu')) 
my_model.add(Dense(1, activation='linear'))

#The first layer (ResNet) of the model is already trained, so we don't need to train it
my_model.layers[0].trainable = False

#Model compilation 
my_model.compile(loss ='mse', optimizer= 'adam', metrics = ['mean_absolute_error']) 
my_model.summary()

#Model fitting 
my_model.fit_generator(train_generator, 
                           steps_per_epoch = 180, 
                           validation_data = val_generator, 
                           validation_steps = 18, 
                           epochs = 30
                      )
#Predictions
pred = my_model.predict_generator(test_generator, verbose = True)
preds_months = pred.flatten

import csv
df_test_temp = pd.read_csv("/kaggle/input/rsna-bone-age/boneage-test-dataset.csv")
image_id = df_test_temp['Case ID']
results=pd.DataFrame({"Image ID":image_id,
                      "Predictions": preds_months})
results.to_csv("predictions.csv",index=False)