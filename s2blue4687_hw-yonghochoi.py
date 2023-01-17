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
import matplotlib as mpl

import matplotlib.pyplot as plt

from IPython.display import display

%matplotlib inline



import pandas as pd

import numpy as np

import cv2

import os

import matplotlib.pyplot as plt

from PIL import Image

from glob import glob

from skimage.feature import hog

from skimage.color import rgb2grey



from sklearn import svm, datasets

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix

from sklearn.model_selection import train_test_split



from sklearn.metrics import roc_curve, auc



from sklearn.preprocessing import RobustScaler
train = list()

label = list()



dataset_train = "/kaggle/input/2019-fall-pr-project/train/train/"

train_data_list = glob(dataset_train+'*.*.jpg')



for i in range(len(train_data_list)):

    if train_data_list[i][47] == 'c':

        label.append(0)

    else :

        label.append(1)

    image = Image.open(train_data_list[i])

    image = image.resize((32,32), Image.ANTIALIAS)

    image = np.array(image)/255

    image = image.reshape(-1)

    train.append(image)
scaler = RobustScaler()



train = np.array(train)

scaler.fit(train)

train = scaler.transform(train)

label = np.array(label)
X_train, X_test, y_train, y_test = train_test_split(train, label, test_size=0.25, random_state=42)
grid_para= {'C': [1, 5, 10, 50], 

            'gamma': [ 0.01,0.05,0.001] ,'class_weight' : ['balanced']}

svc = svm.SVC(kernel = 'rbf',gamma="scale")

clf = GridSearchCV(svc, grid_para, cv=2)
clf.fit(X_train, y_train)

y_predict = clf.predict(X_test)

print(confusion_matrix(y_test, y_predict))

print(classification_report(y_test, y_predict))
test = list()



dataset_test = "/kaggle/input/2019-fall-pr-project/test1/test1/"

test_data_list = glob(dataset_test+'*.jpg')



for i in range(len(test_data_list)):

  image = Image.open(test_data_list[i])

  image = image.resize((32,32), Image.ANTIALIAS)

  image = np.array(image)/255

  image = image.reshape(-1)

  test.append(image)



test = np.array(test)

scaler.fit(test)

test = scaler.transform(test)
result = clf.predict(test)

test = result.reshape(-1,1)