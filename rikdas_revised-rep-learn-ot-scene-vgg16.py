# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
from glob import glob

imagePatches = glob('../input/otscene database/**/*.jpg', recursive=False)

for filename in imagePatches[0:]:

    print(filename)
from keras.layers import Input, Dense

from keras.preprocessing import image

from keras.applications.vgg16 import VGG16

from keras.applications.vgg16 import preprocess_input



model = VGG16(weights='imagenet', include_top=False)



from keras.models import Model

base_model = VGG16(weights='imagenet')#, include_top=True, input_tensor=Input(shape=(128, 128, 3)))

model = Model(inputs=[base_model.input], outputs=[base_model.get_layer('fc2').output])



model.summary()





from glob import glob

imagePatches = glob('../input/otscene database/**/*.jpg', recursive=True)

#imagenumbers = []

#imagenumbers.append(imagePatches)

#imagePatches[:100]



features = []

imagenumber = []

for img in imagePatches:

        imagenumber.append(img)

        img_data = image.load_img(img, target_size=(224, 224))

        img_data = image.img_to_array(img_data)

        img_data = np.expand_dims(img_data, axis=0)

        img_data = preprocess_input(img_data)

        feats = model.predict(img_data)

        features.append(feats.flatten())



vgg16_feature_list_np = np.array(features)

features_ot_scene = pd.DataFrame(vgg16_feature_list_np)

nmbr = np.array(imagenumber)

img_no = pd.DataFrame(nmbr)

img_no.to_csv('img_no_vgg_ot_scene.csv',index=False)

features_ot_scene.to_csv('vgg_16_vgg_ot_scene.csv',index=False) 
features_ot_scene
from sklearn.preprocessing import MinMaxScaler

data = vgg16_feature_list_np

scaler = MinMaxScaler()

scaler.fit(data)

features_norm_ot_scene = scaler.transform(data)

normalized_ot_scene_feature = pd.DataFrame(features_norm_ot_scene)
normalized_ot_scene_feature

normalized_ot_scene_feature.to_csv('normalized_vgg_16_ot_scene.csv',index=False) 