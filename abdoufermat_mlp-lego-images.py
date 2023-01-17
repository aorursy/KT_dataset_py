# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
import sklearn
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import cv2
import os
DATASET_PATH = '/kaggle/input/lego-brick-images/LEGO brick images v1'
!ls -l '/kaggle/input/lego-brick-images/LEGO brick images v1'
def load_images(path):
    images_train = []
    labels_train = []
    
    for sub_folder in os.listdir(path):
        fullpath = os.path.join(path, sub_folder)
        print(fullpath)
        
        if not os.path.isdir(fullpath):
            continue
        images = os.listdir(fullpath)
        
        for image_filename in images:
            image_fullpath = os.path.join(fullpath, image_filename)
            if os.path.isdir(image_fullpath):
                continue
            img = cv2.imread(image_fullpath)
            
            images_train.append(img)
            labels_train.append(sub_folder)
    return np.array(images_train), np.array(labels_train)
images_train, labels_train = load_images(DATASET_PATH)
images_train.shape
labels_train.shape
plt.imshow(images_train[1000])
print('label: ', labels_train[1000])
X = images_train.reshape(images_train.shape[0], images_train.shape[1]*images_train.shape[2]*images_train.shape[3])
X.shape
from sklearn.preprocessing import LabelEncoder

Y = LabelEncoder().fit_transform(labels_train)
np.unique(Y)
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X, Y, shuffle=True, test_size=0.2)
x_train.shape, y_train.shape
from sklearn.neural_network import MLPClassifier

mlp_classifieur = MLPClassifier(activation='relu',
                               hidden_layer_sizes=(100, 100, 100),
                               solver='adam',
                               verbose=True,
                               max_iter=100)
mlp_classifieur.fit(x_train, y_train)
y_pred = mlp_classifieur.predict(x_test)
print('training score: ', mlp_classifieur.score(x_train, y_train))
from sklearn.metrics import accurate_score

print('test score: ', accurate_score(y_test, y_pred))
result = pd.DataFrame({'predicted': y_pred,
                      'reality': y_test})
result.sample(15)
print('SAYONARA')
