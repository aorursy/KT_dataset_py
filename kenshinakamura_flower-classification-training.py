# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

#for dirname, _, filenames in os.walk('/kaggle/input'):

#    for filename in filenames:

#        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
img_paths = []

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        if os.path.splitext(filename)[1] == '.jpg':

            img_paths.append(os.path.join(dirname, filename))



print("イメージ件数：" + str(len(img_paths)))
## 写真をシャッフルする

from random import shuffle

shuffle(img_paths)
img_paths = img_paths[0:100]

print("減らしたイメージ件数：" + str(len(img_paths)))
from tensorflow.python.keras.preprocessing.image import load_img, img_to_array

from keras.applications.resnet50 import ResNet50, preprocess_input

from learntools.deep_learning.decode_predictions import decode_predictions



weights_file='../input/resnet50/resnet50_weights_tf_dim_ordering_tf_kernels.h5'

class_list_path_file='../input/resnet50/imagenet_class_index.json'

model = ResNet50(weights=weights_file)



#include_top=False : 出力層を結合層を除外

#model = ResNet50(weights=weights_file, include_top=False)

model.summary()
## ImageNetイメージサイズに合わせて224x224にする必要あり。

image_size = 224



def read_prep_images(img_path):

    img = load_img(img_path, target_size=(image_size, image_size))

    img = img_to_array(img)

    img = np.expand_dims(img, axis=0)

    output = preprocess_input(img)

    return(output)
def pred_decode(model, img):

    pred = model.predict(img)

    decode = decode_predictions(pred, top=3, class_list_path=class_list_path_file)

    return(decode)
preds_resnet50 = []

for i, img in enumerate(img_paths):

    img = read_prep_images(img)

    preds_resnet50.append(pred_decode(model, img))
from keras.applications.vgg16 import VGG16, preprocess_input



weights_file='../input/vgg16/vgg16_weights_tf_dim_ordering_tf_kernels.h5'

class_list_path_file='../input/vgg16/imagenet_class_index.json'

model = VGG16(weights=weights_file)
model.summary()
preds_vgg16 = []

for i, img in enumerate(img_paths):

    img = read_prep_images(img)

    preds_vgg16.append(pred_decode(model, img))
from IPython.display import Image, display



for i in range(20):

    display(Image(img_paths[i]))

    print("ResNet50:", preds_resnet50[i])

    print("VGG16   :", preds_vgg16[i])