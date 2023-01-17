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
import os

count = 0

d = "/kaggle/input/covid19-radiography-database/COVID-19 Radiography Database/COVID-19"

for path in os.listdir(d):

    if os.path.isfile(os.path.join(d, path)):

        count += 1

print (count)

#OUTPUT
!pip install -U efficientnet
from efficientnet.keras import EfficientNetB7

from keras.preprocessing import image

from efficientnet.keras import preprocess_input

from keras.models import Model

import numpy as np



base_model = EfficientNetB7(weights='imagenet')

#base_model.summary()

model = Model(inputs=[base_model.input], outputs=[base_model.get_layer('probs').output])

model.summary()
#os.mkdir('/kaggle/working/rsna_converted_stage_2_train_images/')

#os.mkdir('/kaggle/working/rsna_converted_stage_2_test_images/')
#import pydicom as dicom

#import os

#import cv2

#import PIL # optional

# make it True if you want in PNG format

#PNG = False

# Specify the .dcm folder path

#folder_path = "/kaggle/input/rsna-pneumonia-detection-challenge/stage_2_train_images/"

# Specify the output jpg/png folder path

#jpg_folder_path = "/kaggle/working/rsna_converted_stage_2_train_images/"

#images_path = os.listdir(folder_path)

#for n, image in enumerate(images_path):

 #   ds = dicom.dcmread(os.path.join(folder_path, image))

  #  pixel_array_numpy = ds.pixel_array

   # if PNG == False:

    #    image = image.replace('.dcm', '.jpg')

    #else:

     #   image = image.replace('.dcm', '.png')

    #cv2.imwrite(os.path.join(jpg_folder_path, image), pixel_array_numpy)

    #if n % 50 == 0:

     #   print('{} image converted'.format(n))
from glob import glob

imagePatches = glob('/kaggle/input/covid19-radiography-database/COVID-19 Radiography Database/COVID-19/*.png', recursive=True)#imagePatches = glob('/kaggle/input/chest-xray-pneumonia/chest_xray/train/NORMAL/*.jpeg', recursive=True)



features = []

imagenumber = []

for img in imagePatches:

        imagenumber.append(img)

        img_data = image.load_img(img, target_size=(600, 600))

        img_data = image.img_to_array(img_data)

        img_data = np.expand_dims(img_data, axis=0)

        img_data = preprocess_input(img_data)

        feats = model.predict(img_data)

        features.append(feats.flatten())

feature = np.array(features)

features_deep = pd.DataFrame(feature)

nmbr = np.array(imagenumber)

img_no = pd.DataFrame(nmbr)

img_no.to_csv('img_no_covid_png.csv',index=False)
#imagePatches
features_deep
from sklearn.preprocessing import MinMaxScaler

data = feature

scaler = MinMaxScaler()

scaler.fit(data)

features = scaler.transform(data)

normalized = pd.DataFrame(features)
normalized
normalized.to_csv('normalized_covid_png.csv',index=False) 