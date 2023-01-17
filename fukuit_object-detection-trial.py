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
bboxlist = pd.read_csv('../input/BBox_List_2017.csv')
bboxlist = bboxlist.rename(columns={'Bbox [x': 'x', 'h]': 'h'})
bboxlist = bboxlist.dropna(axis=1)
# create image list(filepath)
image_list = {}
for root_d, _, files in os.walk('../input/'):
    for f in files:
        if f.endswith('.png') or f.endswith('.jpg'):
            image_list[f] = os.path.join(root_d, f)
# add filepath to bboxlist
bboxlist['Path'] = bboxlist['Image Index'].map(image_list.get)
bboxlist.sample(5)
import cv2
import matplotlib.pyplot as plt

sampleidx = np.random.choice(len(bboxlist), 3, replace=False)
for idx, imgidx in enumerate(sampleidx):
    img = cv2.imread(bboxlist.iloc[imgidx, 6])
    x1 = int(bboxlist.loc[imgidx, 'x'])
    y1 = int(bboxlist.loc[imgidx, 'y'])
    x2 = int(bboxlist.loc[imgidx, 'x'] + bboxlist.loc[imgidx, 'w'])
    y2 = int(bboxlist.loc[imgidx, 'y'] + bboxlist.loc[imgidx, 'h'])

    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
    plt.subplot(1, 3, idx+1)
    plt.imshow(img), plt.title(bboxlist.loc[imgidx, 'Image Index'] + '\n' + bboxlist.loc[imgidx, 'Finding Label'])
    plt.xticks([]), plt.yticks([])
    print('{0}:{1}:(({2},{3}), ({4},{5}))'.format(bboxlist.loc[imgidx, 'Image Index'],
                                                 bboxlist.loc[imgidx, 'Finding Label'],
                                                 x1, y1, x2, y2))
plt.show()
with open('kitti_simple_label.txt', 'w') as f:
    for key, row in bboxlist.iterrows():
        filepath = row['Path']
        x1 = row['x']
        y1 = row['y']
        x2 = row['x']+row['w']
        y2 = row['y']+row['h']
        classname = row['Finding Label']
        f.write('{0:s},{1:f},{2:f},{3:f},{4:f},{5:s}\n'.format(filepath, x1, y1, x2, y2, classname))
from keras.applications import ResNet50
from keras.layers import Input,GlobalAveragePooling2D,Dense,Lambda
from keras.models import Model
from keras.optimizers import adam
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.resnet50 import preprocess_input
from keras.utils.np_utils import to_categorical
inputs = Input(shape=(224, 224, 3))
base_model = ResNet50(include_top=False)
conv = base_model(inputs)
GAV = GlobalAveragePooling2D()(conv)
outputs = Dense(8, activation='softmax')(GAV)
model = Model(inputs, outputs)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

