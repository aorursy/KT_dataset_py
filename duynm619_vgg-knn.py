import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import random

from tqdm import tqdm

import xgboost as xgb

import tensorflow as tf

from keras.applications.resnet50 import ResNet50

# from keras.applications.resnet18 import ResNet18

from keras.models import Model

from keras.preprocessing import image

from keras.applications.resnet50 import preprocess_input, decode_predictions

from keras.layers import Flatten, Input

import scipy

from sklearn.metrics import fbeta_score

from keras.applications.vgg16 import VGG16

from keras.preprocessing import image

from keras.applications.vgg16 import preprocess_input

import numpy as np

from sklearn import preprocessing

from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import train_test_split

import xgboost
data_path = "../input/fruits/data/"



train = os.path.join(data_path, r'train')



train_images = sorted(os.listdir(train))

print("Total number of images in the training set: ", len(train_images))
filenames = os.listdir('../input/fruits/data/train/')

files = []

categories = []

for file in filenames:

    category = file.split('_')[0]

    files.append('../input/fruits/data/train/' + file)

    categories.append(category)

        

        

df = pd.DataFrame({

    'filename': files,

    'category': categories

})
df.head()
df['category'].value_counts()
df.shape
le = preprocessing.LabelEncoder()

le.fit(df['category'])
y = le.transform(df['category'])
base_model = VGG16(weights='imagenet', include_top=False)

inputs = Input(shape=(48,48,3),name = 'image_input')

x = base_model(inputs)

x = Flatten()(x)

model = Model(inputs=inputs, outputs=x)
import time

start = time.time()



x_train = []

y_train = []



for f in tqdm(df.filename[:]):

    img_path = f

    img = image.load_img(img_path, target_size=(48, 48))

    x = image.img_to_array(img)

    x = np.expand_dims(x, axis=0)

    x = preprocess_input(x)



    features = model.predict(x)

    features_reduce =  features.squeeze()

    x_train.append(features_reduce)
x_train = pd.DataFrame(x_train)
x_train.head()
scaler = MinMaxScaler()

scaler.fit(x_train)

x_train = scaler.transform(x_train)
x_train.shape
x_train, x_valid, y_train, y_valid = train_test_split(x_train, y, test_size = 0.2, stratify = y, random_state = 8)
from sklearn import neighbors

clf = neighbors.KNeighborsClassifier(n_neighbors = 1, p = 1)

clf.fit(x_train, y_train)
y_pred = clf.predict(x_valid)
from sklearn.metrics import accuracy_score

from sklearn.metrics import f1_score

print(accuracy_score(y_pred, y_valid))

print(f1_score(y_pred, y_valid, average="macro"))
import seaborn as sn

from sklearn.metrics import classification_report, confusion_matrix

print('Confusion Matrix')

cm = confusion_matrix(y_pred, y_valid)

print(cm)

sn.set(font_scale=1.4)#for label size

sn.heatmap(cm, annot=True,annot_kws={"size": 16})# font size
data_path = "../input/fruits/data/"

test = os.path.join(data_path, r'test')

test_images = sorted(os.listdir(test))

print("Total number of images in the test set: ", len(test_images))
filenames = os.listdir('../input/fruits/data/test/')

files = []

categories = []

for file in filenames:

    category = file.split('_')[0]

    files.append('../input/fruits/data/test/' + file)

    categories.append(category)

        

        

df = pd.DataFrame({

    'filename': files,

    'category': categories

})
df.head()
df['category'].value_counts()
df.shape
le = preprocessing.LabelEncoder()

le.fit(df['category'])
y = le.transform(df['category'])
start = time.time()



x_test = []

y_test = y



for f in tqdm(df.filename[:]):

    img_path = f

    img = image.load_img(img_path, target_size=(48, 48))

    x = image.img_to_array(img)

    x = np.expand_dims(x, axis=0)

    x = preprocess_input(x)



    features = model.predict(x)

    features_reduce =  features.squeeze()

    x_test.append(features_reduce)
x_test = pd.DataFrame(x_test)
x_test.head()
scaler = MinMaxScaler()

scaler.fit(x_train)

x_test = scaler.transform(x_test)
x_test.shape
y_pred = clf.predict(x_test)
print(accuracy_score(y_pred, y_test))

print(f1_score(y_pred, y_test, average="macro"))