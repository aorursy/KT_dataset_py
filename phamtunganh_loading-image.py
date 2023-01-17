# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input/album_girl"))



# Any results you write to the current directory are saved as output.
!pip install --upgrade imutils

from imutils import paths

import random

# Lấy các đường dẫn đến ảnh.

#print(os.listdir("../input/dataset/dataset"))

image_path = list(paths.list_images('../input/album_girl'))

#image_path = list(paths.list_images('../input/dogandcat/dogandcat/dogandcat/train'))

print(image_path[0])



# Đổi vị trí ngẫu nhiên các đường dẫn ảnh

random.shuffle(image_path)
from keras.preprocessing.image import img_to_array

from keras.preprocessing.image import load_img

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import LabelBinarizer



# Đường dẫn ảnh sẽ là dataset/tên_loài_hoa/tên_ảnh ví dụ dataset/Bluebell/image_0241.jpg nên p.split(os.path.sep)[-2] sẽ lấy ra được tên loài hoa

labelsName = [p.split(os.path.sep)[-2] for p in image_path]



# Chuyển tên các loài hoa thành số

le = LabelEncoder()

labels = le.fit_transform(labelsName)



# One-hot encoding

lb = LabelBinarizer()

labels = lb.fit_transform(labels)
from keras.applications import imagenet_utils

# Load ảnh và resize về đúng kích thước mà VGG 16 cần là (224,224)

list_image = []

for (j, imagePath) in enumerate(image_path):

    image = load_img(imagePath, target_size=(224, 224))

    image = img_to_array(image)

    

    image = np.expand_dims(image, 0)

    image = imagenet_utils.preprocess_input(image)

    

    list_image.append(image)

    

list_image = np.vstack(list_image)