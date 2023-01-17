# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

from sklearn.model_selection import train_test_split
from sklearn import svm
import matplotlib.pyplot as plt, matplotlib.image as mpimg

# Any results you write to the current directory are saved as output.
z= pd.read_csv("../input/train.csv") #labelled images set
x=z.iloc[:500,1:] #images only
y=z.iloc[:500,:1] # label of the images

train_x,test_x,train_y,test_y= train_test_split(x,y,train_size=0.8)


# img= train_x.as_matrix()   #sample image to display is converted from frame to array
# img= img.reshape((28,28))  #converting array into matrix

# plt.imshow(img,cmap='gray')
# plt.title(train_y.iloc[4])



test_x[test_x>0]=1
train_x[train_x>0]=1


csf=svm.SVC() #calling classifier
csf.fit(train_x,train_y.values.ravel())
csf.score(test_x,test_y)
test_data=pd.read_csv('../input/test.csv')
test_data[test_data>0]=1
results=csf.predict(test_data[0:500])
results
