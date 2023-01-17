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

from sklearn.preprocessing import scale

!pip install imutils

import imutils

import numpy as np

import cv2 

import os



#os.listdir("/kaggle/input/2019-fall-pr-project/")



dataset_train = "/kaggle/input/2019-fall-pr-project/train/train"

imgs=[]

labels=[]

#import pdb;pdb.set_trace()

for i in os.listdir(dataset_train):

 

  img=cv2.imread(os.path.join(dataset_train,i))

  img.resize([32,32,3])

  #img=np.transpose(img,(2,0,1))

  img=img.reshape([3*32*32])



  imgs.append(img) 

  label=i.split('.')[0]

  if label=='cat':

    label=0

  elif label=='dog':

    label=1

  labels.append(label)

  #import pdb;pdb.set_trace()

  

labels=np.array(labels)

imgs=np.array(imgs)
from sklearn import preprocessing

nomal=preprocessing.MinMaxScaler()



X_train, X_val, y_train, y_val=train_test_split(imgs,labels,test_size=0.9)

X_train=scale(X_train)

X_val=scale(X_val)
X_train.shape
from sklearn.model_selection import GridSearchCV

from sklearn.svm import SVC

from sklearn.metrics import classification_report

param_grid = {'C': [1e3,  1e4, 1e5],

              'gamma': [ 0.0005,  0.005,  0.1], }

clf = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'),

                   param_grid, cv=5, iid=False,verbose=1)

SVM=clf.fit(X_train, y_train)

#clf = clf.fit(X_train, y_train)

label=SVM.predict(X_val)



print(classification_report(y_val,label))


dataset_test = "/kaggle/input/2019-fall-pr-project/test1/test1"







imgstest=[]

labelstest=[]

ids=[]

for i in os.listdir(dataset_test):

  #import pdb;pdb.set_trace()

  ids.append(int(i.split('.')[0]))

  img=cv2.imread(os.path.join(dataset_test,i))

  img.resize([32,32,3])

  #img=np.transpose(img,(2,0,1))

  img=img.reshape([3*32*32])



  imgstest.append(img) 







imgstest=np.array(imgstest)



labelstest=np.array(labelstest)



result=SVM.predict(imgstest)








