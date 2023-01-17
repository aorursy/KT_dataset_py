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

        #print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import cv2
from IPython.display import Image, display

import numpy as np

import os

from os.path import join

from PIL import ImageFile

import pandas as pd

from matplotlib import cm

import seaborn as sns

from sklearn.metrics import mean_squared_error, mean_absolute_error, roc_auc_score, classification_report, confusion_matrix

import matplotlib.pyplot as plt

from sklearn.utils import shuffle

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA

from sklearn.ensemble import IsolationForest

from sklearn import svm

from sklearn.mixture import GaussianMixture

from sklearn.isotonic import IsotonicRegression

import re



ImageFile.LOAD_TRUNCATED_IMAGES = True

plt.style.use('fivethirtyeight')

%matplotlib inline

# import car images from natural images
# import car images from natural images

train_img_dir_0 = "../input/natural-images/data/natural_images/car" 

train_img_dir_1 = "../input/natural-images/data/natural_images/person"

train_img_dir_2 = "../input/natural-images/data/natural_images/motorbike" 

train_img_paths_0= [join(train_img_dir_0,filename) for filename in os.listdir(train_img_dir_0)]

train_img_paths_1 = [join(train_img_dir_1,filename) for filename in os.listdir(train_img_dir_1)]

train_img_paths_2 = [join(train_img_dir_2,filename) for filename in os.listdir(train_img_dir_2)]
train_img_paths_n=train_img_paths_0 + train_img_paths_1 +train_img_paths_2
labels=[]

for i in train_img_paths_n:

    l=(i.split('_')[1].split('/')[2])

    if(l=='car'):

        labels.append(0)

    elif(l=='person'):

        labels.append(1)

    else:

        labels.append(2)
features=[]

loc0="../input/natural-images/data/natural_images/car"

loc1="../input/natural-images/data/natural_images/person"

loc2="../input/natural-images/data/natural_images/motorbike"#repeatedly for car then person then motorbike

for i in os.listdir(loc0):

    iml=os.path.join(loc0,i)

    f=cv2.imread(iml,0)

    fr=cv2.resize(f,(100,100))    

    features.append(fr)



for i in os.listdir(loc1):

    iml=os.path.join(loc1,i)

    f=cv2.imread(iml,0)

    fr=cv2.resize(f,(100,100))    

    features.append(fr)

    

for i in os.listdir(loc2):

    iml=os.path.join(loc2,i)

    f=cv2.imread(iml,0)

    fr=cv2.resize(f,(100,100))    

    features.append(fr)
np.array(features).shape
X=np.array(features).reshape(-1,10000)

y=labels
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3)
from sklearn.tree import DecisionTreeClassifier
dmodel=DecisionTreeClassifier()
dmodel.fit(X_train,y_train)
y_pred=dmodel.predict(X_test)
dmodel.score(X_test,y_test)