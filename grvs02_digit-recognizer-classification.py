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
import pandas as pd
import matplotlib.pyplot as plt, matplotlib.image as mpimg
from sklearn.model_selection import train_test_split
from sklearn import svm
%matplotlib inline


l_image = pd.read_csv('../input/train.csv')
l_image.info()
images_l = l_image.iloc[0:5000, 1:]
labels_l = l_image.iloc[0:5000, :1]
images_l.shape
     #just for knowledge
labels_l.shape #just for knowledge
#creating train and test sets

train_images, test_images, train_labels, test_labels = train_test_split(images_l,labels_l,train_size=0.8, random_state = 0)
#convert from series(1 d array to 2d array)

i=1
img = train_images.iloc[i].as_matrix()
img = img.reshape((28,28))
plt.imshow(img, cmap = 'gray')
plt.title(train_labels.iloc[i,0])


plt.hist(train_images.iloc[i])
#training our model

clf = svm.SVC()
clf.fit(train_images,train_labels.values.ravel())
clf.score(train_images,test_labels)
#test data prediction
test_data = pd.read_csv('../input/test.csv')
test_data[test_data>0] = 1
results=clf.predict(test_data[0:5000])
