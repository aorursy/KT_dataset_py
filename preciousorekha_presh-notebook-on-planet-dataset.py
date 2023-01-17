# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

#import numpy as np # linear algebra
#import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

#import os
#for dirname, _, filenames in os.walk('/kaggle/input'):
#    for filename in filenames:
       # print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# importing relevant libraries

import os
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.metrics import fbeta_score
from tqdm import tqdm
import cv2
from PIL import Image
from tensorflow import keras
from skimage import io
from sklearn.preprocessing import MultiLabelBinarizer
# loading the csv metadata files

train_classes = pd.read_csv("../input/planets-dataset/planet/planet/train_classes.csv")
sample_sub = pd.read_csv("../input/planets-dataset/planet/planet/sample_submission.csv")
train_classes.head()
# dict for converting labels to numerical classes

label_map = {'agriculture': 14,
 'artisinal_mine': 5,
 'bare_ground': 1,
 'blooming': 3,
 'blow_down': 0,
 'clear': 10,
 'cloudy': 16,
 'conventional_mine': 2,
 'cultivation': 4,
 'habitation': 9,
 'haze': 6,
 'partly_cloudy': 13,
 'primary': 7,
 'road': 11,
 'selective_logging': 12,
 'slash_burn': 8,
 'water': 15}
# Loading the training images

#x_train, y_train = [], []

#for img_name, tags in tqdm(train_classes.values, miniters=1000):
    #arr = cv2.imread('../input/planets-dataset/planet/planet/train-jpg/{}.jpg'.format(img_name))
    #targets = np.zeros(17)
    #for t in tags.split(' '):
   #     targets[label_map[t]] = 1 
  #  x_train.append(cv2.resize(arr, (64, 64)))
 #   y_train.append(targets)

# normalizing train image pixels
#y_train = np.array(y_train, np.uint8)
#x_train = np.array(x_train,np.float16)/255.0
# numbers of tags and their names
counts = {}
splitted_tags = train_classes['tags'].map(lambda x: x.split(' '))
for labels in splitted_tags.values:
    for label in labels:
        counts[label] = counts[label] + 1  if label in counts else 0

plt.figure(figsize=(18, 6))
plt.title('Classes')
idxs = range(len(counts.values()))
plt.xticks(idxs, counts.keys(), rotation=-45)
plt.bar(idxs, counts.values());
len(splitted_tags)
#load data
all_labels = splitted_tags.values
labels = list(set([y for x in all_labels for y in x]))

def load_data(train_classes, labels, resize):
    x_train = []
    y_train = []

    label_map = {l: i for i, l in enumerate(labels)}
    inv_label_map = {i: l for l, i in label_map.items()}

    for f, tags in train_classes.values:
        img = cv2.imread('../input/planets-dataset/planet/planet/train-jpg/{}.jpg'.format(f)) 
        targets = np.zeros(17)
        for t in tags.split(' '):
            targets[label_map[t]] = 1 

        x_train.append(cv2.resize(img,resize))
        y_train.append(targets)
        
    y_train = np.array(y_train, np.uint8)
    x_train = np.array(x_train, np.float16) / 255.

    return x_train, y_train
import gc
gc.collect()
x, y = load_data(train_classes, labels, resize=(64, 64))
gc.collect()
print(x.shape)
print(y.shape)
# checking the images of the datasets

print(train_classes.shape)
print(sample_sub.shape)
from sklearn.model_selection import train_test_split

import time
x_train, x_val, y_train, y_val = train_test_split(x,y, test_size=0.2, random_state = int(time.time()))
print(y_train.shape)
print(x_train.shape)
print(x_val.shape)
print(y_val.shape)
# making use of the training set


#img='train_10016.jpg'
#path = '../input/planets-dataset/planet/planet/train-jpg/{}'.format(img)

#plt.imshow(io.imread(path))

#
%reload_ext autoreload
%autoreload 2
%matplotlib inline

from fastai.vision import *
tfms = get_transforms(flip_vert=True, max_lighting=0.1, max_zoom=1.05, max_warp=0.)



path = Path('/kaggle/input/planets-dataset/planet/planet')
path.ls()
np.random.seed(42)
src = (ImageList.from_csv(path,'train_classes.csv', folder='train-jpg', suffix='.jpg')
       .split_by_rand_pct(0.2)
       .label_from_df(label_delim=' '))


data = (src.transform(tfms, size=128)
        .databunch(num_workers=0).normalize(imagenet_stats))

data.show_batch(rows=3, figsize=(12,9))
# Since this is a multi lable task and the labels are given as tags in a single dataframe series

biner = MultiLabelBinarizer()
tags = train_classes['tags'].str.split()
y = biner.fit_transform(tags)

labels = biner.classes_
print('Number of labels: ', len(labels))
print('\n')
print(labels)
# Getting the labels into one hot encoded form for EDA ease. 

#for label in labels:
    #train_classes[label] = train_classes['tags'].apply(lambda x: 1 if label in x.split()  else 0)
    
#train_classes.head()
#train_classes[labels].sum().sort_values(ascending=False).plot(kind='barh', figsize=(8,8))



def learning_curve(model_fit, key='acc', ylim=(0.8, 1.01)):
    plt.figure(figsize=(12,6))
    plt.plot(model_fit.history[key])
    plt.plot(model_fit.history['val_' + key])
    plt.title('Learning Curve')
    plt.ylabel(key.title())
    plt.xlabel('Epoch')
    plt.ylim(ylim)
    plt.legend(['train', 'test'], loc='best')
    plt.show()




def fbeta_score_K(y_true, y_pred):
    beta_squared = 4

    tp = K.sum(y_true * y_pred) + K.epsilon()
    fp = K.sum(y_pred) - tp
    fn = K.sum(y_true) - tp

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    result = (beta_squared + 1) * (precision * recall) / (beta_squared * precision + recall + K.epsilon())
    return result


from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPool2D
from keras.applications import ResNet50, VGG16
from keras.optimizers import Adam
gc.collect()
optimizer = Adam(0.003, decay=0.0005)
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(64, 64, 3))

for layer in base_model.layers:
    layer.trainable = False
    
    model = Sequential([
    base_model,
 
    Flatten(), 
        
    Dense(128, activation='relu'),
    Dropout(0.2),
    Dense(17, activation='sigmoid')  
])

model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=[fbeta_score_K])
model.summary()
model_fit = model.fit( x_train, y_train, batch_size=64, epochs=10, verbose=1, validation_data=(x_val, y_val))
y_pred = model.predict(x_val, batch_size=64)
score = fbeta_score(y_val, np.array(y_pred) > 0.2, beta=2, average='samples')

print("Test score (f1): ", score)
print("Error: %.2f%%" % (100-score*100))
learning_curve(model_fit, key='loss', ylim=(0, 1))
gc.collect()
# decrease learning step and decay

optimizer = Adam(0.0001, decay=0.00001)
model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=[fbeta_score_K])

model_fit = model.fit(
    x_train, y_train,
    batch_size=64,
    epochs=10,
    verbose=1,
    validation_data=(x_val, y_val))
y_pred = model.predict(x_val, batch_size=64)
score = fbeta_score(y_val, np.array(y_pred) > 0.2, beta=2, average='samples')

print("Test score (f1): ", score)
print("Error: %.2f%%" % (100-score*100))
learning_curve(model_fit, key='loss', ylim=(0, 1))
gc.collect()
# adding more layer to learn

for layer in model.layers[0].layers[1:]:
    layer.trainable = True

for layer in model.layers[0].layers:
    print(layer.name, layer. trainable)

model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=[fbeta_score_K])
model.summary()
model_fit = model.fit(
    x, y,
    batch_size=64,
    epochs=20,
    verbose=1,
    validation_data=(x_val, y_val))
y_pred = model.predict(x_val, batch_size=64)
score = fbeta_score(y_val, np.array(y_pred) > 0.2, beta=2, average='samples')

print("F beta score: ", score)
print("Error: %.2f%%" % (100-score*100))
gc.collect()
learning_curve(model_fit, key='loss', ylim=(0, 1))
# I will check fit_generator for my the best solution

aug = keras.preprocessing.image.ImageDataGenerator(rotation_range=20, zoom_range=0.15,
                         width_shift_range=0.2, height_shift_range=0.2, shear_range=0.15,
                         horizontal_flip=True, fill_mode="nearest")
 
model_fit = model.fit_generator(aug.flow(x, y, batch_size=64),
                        validation_data=(x_val, y_val), steps_per_epoch=len(x) // 128,
                        epochs=5)
y_pred = model.predict(x_val, batch_size=64)
score = fbeta_score(y_val, np.array(y_pred) > 0.2, beta=2, average='samples')

print("F beta score: ", score)
print("Error: %.2f%%" % (100-score*100))
learning_curve(model_fit, key='loss', ylim=(0, 1))
# now to check the Test data


X_test=[]

for img, label in tqdm(sample_sub[:40669].values, miniters = 1000):
  X_test.append(cv2.resize(cv2.imread('../input/planets-dataset/planet/planet/test-jpg/{}.jpg'.format(img)), (64,64)))

for img, label in tqdm(sample_sub[40669:].values, miniters = 1000):
  X_test.append(cv2.resize(cv2.imread('../input/planets-dataset/test-jpg-additional/test-jpg-additional/{}.jpg'.format(img)), (64,64)))

x_test = np.array(X_test, np.float16)/255
x_test.shape
Test_Predictions = model.predict(x_test, batch_size = 64)
Test_Predictions

pred = pd.DataFrame(Test_Predictions, columns= labels)
pred
labels
# kaggle submission



learning_curve(model_fit, key='loss', ylim=(0, 1))
final_pred = []

for i in tqdm(range(pred.shape[0]), miniters=1000):
    a = pred.loc[[i]]
    a = a.apply(lambda x:x>0.2, axis =1)
    a = a.transpose()
    a = a.loc[a[i]==True]
    ' '.join(list(a.index))
    final_pred.append(' '.join(list(a.index)))
gc.collect()
sample_sub['tags'] = final_pred
sample_sub.to_csv('My_Final_Result.csv', index = False)
My_result
