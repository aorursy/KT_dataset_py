import scipy.io

import numpy as np

import pandas as pd

from datetime import datetime, timedelta

import tensorflow as tf

import keras

from keras.preprocessing import image

from keras.callbacks import ModelCheckpoint,EarlyStopping

from keras.layers import Dense, Activation, Dropout, Flatten, Input, Convolution2D, ZeroPadding2D, MaxPooling2D, Activation

from keras.layers import Conv2D, AveragePooling2D

from keras.models import Model, Sequential

from sklearn.model_selection import train_test_split

from keras import metrics

from keras.models import model_from_json

import matplotlib.pyplot as plt



mat = scipy.io.loadmat('/kaggle/input/wiki-crop/wiki_crop/wiki.mat')
instances = mat['wiki'][0][0][0].shape[1]

 

columns = ["dob", "photo_taken", "full_path", "gender", "name", "face_location", "face_score", "second_face_score"]

 

import pandas as pd

df = pd.DataFrame(index = range(0,instances), columns = columns)

 

for i in mat:

    if i == "wiki":

        current_array = mat[i][0][0]

        for j in range(len(current_array)):

            df[columns[j]] = pd.DataFrame(current_array[j][0])
from datetime import datetime, timedelta

def datenum_to_datetime(datenum):

    days = datenum % 1

    hours = days % 1 * 24

    minutes = hours % 1 * 60

    seconds = minutes % 1 * 60

    exact_date = datetime.fromordinal(int(datenum)) \

    + timedelta(days=int(days)) + timedelta(hours=int(hours)) \

    + timedelta(minutes=int(minutes)) + timedelta(seconds=round(seconds)) \

    - timedelta(days=366)



    return exact_date.year

 

df['date_of_birth'] = df['dob'].apply(datenum_to_datetime)
df['age'] = df['photo_taken'] - df['date_of_birth']
#remove pictures does not include face

df = df[df['face_score'] != -np.inf]

 

#some pictures include more than one face, remove them

df = df[df['second_face_score'].isna()]

 

#check threshold

df = df[df['face_score'] >= 3]

 

#some records do not have a gender information

df = df[~df['gender'].isna()]

 

df = df.drop(columns = ['name','face_score','second_face_score','date_of_birth','face_location'])
#some guys seem to be greater than 100. some of these are paintings. remove these old guys

df = df[df['age'] <= 100]

 

#some guys seem to be unborn in the data set

df = df[df['age'] > 0]
histogram_age = df['age'].hist(bins=df['age'].nunique())
classes = 101 #(0, 100])

print("number of output classes: ",classes)
target_size = (224, 224)



def getImagePixels(image_path):

    img = image.load_img("/kaggle/input/wiki-crop/wiki_crop/%s" % image_path[0], grayscale=False, target_size=target_size)

    x = image.img_to_array(img).reshape(1, -1)[0]

    #x = preprocess_input(x)

    return x



df['pixels'] = df['full_path'].apply(getImagePixels)



df.head()
target = df['age'].values

target_classes = keras.utils.to_categorical(target, classes)



#features = df['pixels'].values

features = []



for i in range(0, df.shape[0]):

    features.append(df['pixels'].values[i])



features = np.array(features)

features = features.reshape(features.shape[0], 224, 224, 3)



features.shape