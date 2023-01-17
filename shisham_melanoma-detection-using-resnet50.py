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
import os, random, re, math, time
random.seed(a=42)
import numpy as np

import pandas as pd
import tensorflow as tf
import tensorflow.keras.backend as K
import PIL
from kaggle_datasets import KaggleDatasets
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


## import the dataset
dftrain = pd.read_csv('../input/siim-isic-melanoma-classification/train.csv')
dftest = pd.read_csv('../input/siim-isic-melanoma-classification/train.csv')
## Missing values in training data
dftrain.isna().any()
dftrain = dftrain.dropna()

## Missing values in testing data
dftest.isna().any()
dftest = dftest.dropna()

print("Count of male and female ", dftrain.sex.value_counts())
print("uniques in anatom_site_general_challenge column ", dftrain.anatom_site_general_challenge.unique())
print("uniques in diagnosis column ", dftrain.diagnosis.unique())
print("uniques in benign_malignant column ", dftrain.benign_malignant.unique())
sns.kdeplot(dftrain[(dftrain['target'] == 1)].age_approx, shade = True)
dummy = dftrain[dftrain['target'] == 1]

sns.barplot(x = "sex", y = "target", data = dummy, hue ="sex")


# viewing the distributions of data 

sns.barplot(x = "target", y = "target", data = dftrain, hue ="target")

## hence the number of unaffected surpasses the affected, we need to normalise this data. But we need to understand 
# various other columns before normalising this data
print("no of unaffected males  : ", len(dftrain[(dftrain['target'] == 0) & (dftrain['sex'] == 'male')]))
print("no of unaffected females  : ", len(dftrain[(dftrain['target'] == 0) & (dftrain['sex'] == 'female')]))

## not. much of a difference in. the ratios
sns.barplot(x = "anatom_site_general_challenge", y = "target", data = dftrain, hue ="anatom_site_general_challenge")

## looking at the unique values of anatom_site_general_challenge column

dftrain.anatom_site_general_challenge.unique()
# looking at the number of unique values of anatom_site_general_challenge column

df = dftrain.copy()
print("no of head/neck  : ", len(df[(df['target'] == 0) & (df['anatom_site_general_challenge'] == 'head/neck')]))
print("no of upper extremity  : ", len(df[(df['target'] == 0) & (df['anatom_site_general_challenge'] == 'upper extremity')]))
print("no of lower extremity  : ", len(df[(df['target'] == 0) & (df['anatom_site_general_challenge'] == 'lower extremity')]))
print("no of torso  : ", len(df[(df['target'] == 0) & (df['anatom_site_general_challenge'] == 'torso')]))
print("no of palms/soles'  : ", len(df[(df['target'] == 0) & (df['anatom_site_general_challenge'] == 'palms/soles')]))
print("no of oral/genital'  : ", len(df[(df['target'] == 0) & (df['anatom_site_general_challenge'] == 'oral/genital')]))
## number of affected in the whole dataset

print("no of affected : ", len(dftrain[dftrain['target'] == 1]))
temp = dftrain[dftrain['target'] == 1]

## for training  (500 samples)
final_df = temp[:500]

## for testing (75 samples)
affected_validationdata = temp[500:]
def random_data_selector(data, n):
    data =  data.sample(n = n)
    return data


df = dftrain.copy()

## selecting samples for testing (100 samples)
temp  = df[df['target'] == 0]
unaffected_validationdata = temp[:100]


data = random_data_selector(df[(df['target'] == 0) & (df['anatom_site_general_challenge'] == 'head/neck')], 100)
final_df = final_df.append(data, ignore_index = True)

data = random_data_selector(df[(df['target'] == 0) & (df['anatom_site_general_challenge'] == 'upper extremity')], 100)
final_df = final_df.append(data, ignore_index = True)

data = random_data_selector(df[(df['target'] == 0) & (df['anatom_site_general_challenge'] == 'lower extremity')], 100)
final_df = final_df.append(data, ignore_index = True)

data = random_data_selector(df[(df['target'] == 0) & (df['anatom_site_general_challenge'] == 'torso')], 100)
final_df = final_df.append(data, ignore_index = True)

data = random_data_selector(df[(df['target'] == 0) & (df['anatom_site_general_challenge'] == 'palms/soles')], 100)
final_df = final_df.append(data, ignore_index = True)

data = random_data_selector(df[(df['target'] == 0) & (df['anatom_site_general_challenge'] == 'oral/genital')], 100)
final_df = final_df.append(data, ignore_index = True)


final_df.head()
## finding the number of infected
len(final_df[final_df['target'] == 1])
## finding the number of infected
len(final_df[final_df['target'] == 0])
## shuffle the dataset

import sklearn

final_df  = sklearn.utils.shuffle(final_df)
import cv2
import pathlib
import imageio
from skimage.transform import resize
import numpy as np
from keras.preprocessing import image
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions



#converting normal images into numpy array


training_paths = pathlib.Path('../input/siim-isic-melanoma-classification/jpeg').glob('train/*.jpg')
training_sorted = sorted([x for x in training_paths])
directory_path = '../input/siim-isic-melanoma-classification/jpeg/train/'



for index in range(len(training_sorted)) : training_sorted[index] = str(training_sorted[index])    
    
training_images = np.zeros(150528)
training_images = training_images.reshape(1,224,224,3)

for index in range(len(final_df)):
    img_name = final_df.loc[index].image_name
    img_name = str(directory_path +  img_name + '.jpg')
    position_in_list = training_sorted.index(img_name)
    img_path = training_sorted[position_in_list]
    
    img = image.load_img(img_path, target_size = (224,224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis = 0)
    x = preprocess_input(x)    
    training_images = np.vstack((training_images, x))

    
## making the xtrain data ready

xtraindf = training_images.copy()
xtraindf = xtraindf[1:]
xtraindf.shape

import matplotlib.pyplot as plt
fig, axes = plt.subplots(5,5, figsize=(8,8))

for i,ax in enumerate(axes.flat):
    ax.imshow(xtraindf[i])
## making the ytrain data ready

ytraindf = final_df.target
ytraindf.head()
affected_validationdata = affected_validationdata.append(unaffected_validationdata)
testdata_copy = affected_validationdata.copy()
testdata_copy.head()
print("affected : ", len(testdata_copy[testdata_copy['target'] == 1]))
print("unaffected : ", len(testdata_copy[testdata_copy['target'] == 0]))
## shuffle the df 
import sklearn

testdata_copy  = sklearn.utils.shuffle(testdata_copy).reset_index(drop=True)
testdata_copy.tail()


#converting normal images into numpy array


test_paths = pathlib.Path('../input/siim-isic-melanoma-classification/jpeg/').glob('train/*.jpg')
test_sorted = sorted([x for x in test_paths])
directory_path = '../input/siim-isic-melanoma-classification/jpeg/train/'



for index in range(len(test_sorted)) : test_sorted[index] = str(test_sorted[index])    
    
test_images = np.zeros(150528)
test_images = test_images.reshape(1,224,224,3)

for index in range(len(testdata_copy)):
    img_name = testdata_copy.loc[index].image_name
    img_name = str(directory_path +  img_name + '.jpg')
    
    position_in_list = test_sorted.index(img_name)
    img_path = test_sorted[position_in_list]
    
    img = image.load_img(img_path, target_size = (224,224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis = 0)
    x = preprocess_input(x)    
    test_images = np.vstack((test_images, x))
        

    
## Making the xtest data ready

xtestdf = test_images.copy()
xtestdf = xtestdf[1:]
xtestdf.shape

## plotting the images

import matplotlib.pyplot as plt
fig, axes = plt.subplots(5,5, figsize=(8,8))

for i,ax in enumerate(axes.flat):
    ax.imshow(xtestdf[i])
## making the ytest data ready

ytestdf = testdata_copy.target
ytestdf.head()

from keras.applications.resnet50 import ResNet50

img_rows, img_cols = 224, 224


resnet = ResNet50(weights = 'imagenet',
                      include_top = False,
                      input_shape  =  (img_rows,  img_cols, 3))

## lets freeze the last  4 layers as they  are set to be trainable by  default
for layer in  resnet.layers:
  layer.trainable = False

## lets look at our layers
for (i, layer) in enumerate(resnet.layers):
  print(str(i)  + " "  +  layer.__class__.__name__, layer.trainable)

## Let us  create a function to build our top layer

def addTopMobileNetLayer(bottom_model, num_classes):

  top_model = bottom_model.output
  top_model = GlobalAveragePooling2D()(top_model)
  top_model = (Dense(2048, activation = 'relu'))(top_model)
  top_model = (Dense(2048, activation = 'relu'))(top_model)
  top_model = (Dense(1024, activation = 'relu'))(top_model)
  top_model = (Dense(num_classes, activation  = 'softmax'))(top_model)

  return top_model
from keras.layers import Dense, GlobalAveragePooling2D, Dropout, Activation,  Flatten
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.models import Model

num_classes = 2

FC_head = addTopMobileNetLayer(resnet, num_classes)

model  = Model(inputs =  resnet.input, outputs = FC_head)

print(model.summary())

from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping

earlystop = EarlyStopping(monitor = 'val_loss', 
                          min_delta = 0, 
                          patience = 3,
                          verbose = 1,
                          restore_best_weights = True)

callbacks = [earlystop]


model.compile(loss = 'binary_crossentropy', optimizer = Adam(lr = 0.005), metrics = ['accuracy'])
epochs = 10


history = model.fit(xtraindf, 
          ytraindf,
          batch_size = 8,
          epochs = epochs,
          verbose = 1,
          callbacks = callbacks,
          validation_data = (xtestdf, ytestdf)
          )

import tensorflow
import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from keras.optimizers import Adam, RMSprop
from tensorflow.keras.callbacks import TensorBoard
model = Sequential()
model.add(Conv2D(24,3,3, input_shape = (224,224,3), activation = 'relu'))
model.add(Conv2D(36,3,3, activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Conv2D(36,3,3, activation = 'relu'))
model.add(Conv2D(36,3,3,  activation = 'relu'))

model.add(Flatten())
model.add(Dense(units = 2028, activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(units = 1024, activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(units = 512, activation = 'relu'))
model.add(Dense(units = 1, activation = 'sigmoid'))
model.compile(loss = 'binary_crossentropy', optimizer = RMSprop(lr = 0.005), metrics = ['accuracy'])
epochs = 10


history = model.fit(xtraindf, 
          ytraindf,
          batch_size = 64,
          epochs = epochs,
          verbose = 1,
          validation_data = (xtestdf, ytestdf))
#plot of validation vs training accuracy over the epochs

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['training', 'validation'])
plt.title('Loss')
plt.xlabel('Epoch')
ypred = model.predict(xtestdf)
evaluation = model.evaluate(xtestdf, ytestdf)
print('Test accuracy : {:.3f}'.format(evaluation[1]))
