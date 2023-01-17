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
import pandas as pd 
images_path = "/kaggle/input/chest-xrays-indiana-university/images/images_normalized/"
df  =pd.read_csv("/kaggle/input/chest-xrays-indiana-university/indiana_reports.csv")
df2 = pd.read_csv("/kaggle/input/chest-xrays-indiana-university/indiana_projections.csv")
df2 = df2[df2['projection']=='Frontal']
df  =pd.merge(df,df2,  on=['uid'])

df.head()
files = df['filename']
files.head()
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
i=0
for name in files:
    filename = images_path + '/' + name
    image = load_img(filename, target_size=(224, 224))
        # convert the image pixels to a numpy array
    image = img_to_array(image)
        # reshape data for the model
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        # prepare the image for the VGG model
        # get image id
    image_id = name.split('.')[0]
        # store feature
    print(image_id)
    #features[image_id] = feature
    print('>%s' % name)
    if i==0:
        break
    
from os import listdir
from pickle import dump
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.models import Model


# extract features from each photo in the directory
def extract_features(directory,files):
    # load the model
    model = VGG16(weights=None)
    # re-structure the model
    model.layers.pop()
    model = Model(inputs=model.inputs, outputs=model.layers[-1].output)
    # extract features from each photo
    features = dict()
    for name in files:
        # load an image from file
        filename = directory + '/' + name
        image = load_img(filename, target_size=(224, 224))
        # convert the image pixels to a numpy array
        image = img_to_array(image)
        # reshape data for the model
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        # prepare the image for the VGG model
        image = preprocess_input(image)
        # get features
        feature = model.predict(image, verbose=0)
        # get image id
        image_id = name.split('.')[0]
        # store feature
        features[image_id] = feature
        print('>%s' % name)
    return features
# extract features from all images
directory = images_path
features = extract_features(directory,files)
print('Extracted Features: %d' % len(features))
# save to file
dump(features, open('features.pkl', 'wb'))
!ls