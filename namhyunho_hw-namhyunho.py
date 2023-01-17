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
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import train_test_split

import numpy as np

import cv2

import os

from glob import glob

from sklearn import svm, datasets

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix

from PIL import Image
dataset_train = "/kaggle/input/2019-fall-pr-project/train/train/"

train_data_list = glob(dataset_train+'*.*.jpg')
train = list()

label = list()

imgsize = (32,32)

for i in range(len(train_data_list)):

    if train_data_list[i][47] == 'c':

        label.append(0)

    else :

        label.append(1)

    image = Image.open(train_data_list[i])

    image = image.resize(imgsize, Image.ANTIALIAS)

    image = np.array(image)/255

    image = image.reshape(-1)

    train.append(image)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()



train = np.array(train)

scaler.fit(train)

train = scaler.transform(train)

label = np.array(label)
X_train, X_test, y_train, y_test = train_test_split(train, label, test_size=0.75, random_state=42)
from sklearn.model_selection import GridSearchCV

grid_P= {'C': [1, 10, 100, 1e3], 'gamma': [ 0.1,0.05,0.005] ,'class_weight' : ['balanced']}

svc = svm.SVC(kernel = 'rbf',gamma="scale")

clf = GridSearchCV(svc, grid_P, cv=2)

clf.fit(X_train, y_train)
y_predict = clf.predict(X_test)
test = list()



dataset_test = "/kaggle/input/2019-fall-pr-project/test1/test1/"

test_data_list = glob(dataset_test+'*.jpg')



for i in range(len(test_data_list)):

  image = Image.open(test_data_list[i])

  image = image.resize(imgsize, Image.ANTIALIAS)

  image = np.array(image)/255

  image = image.reshape(-1)

  test.append(image)



test = np.array(test)

scaler.fit(test)

test = scaler.transform(test)
result = clf.predict(test)
test = result.reshape(-1,1)


import pandas as pd



print(test.shape)

df = pd.DataFrame(test, columns=["label"])

df.index = np.arange(1,len(df)+1)

df.index.name = 'id'



df.to_csv('results-hhnam-v.csv',index=True, header=True)