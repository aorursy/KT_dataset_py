# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
ex6data1 = pd.read_csv("../input/ex6data1.csv",header=None)
ex6data1.head()
ex6data1.plot.scatter(x=0,y=1,c=ex6data1[2].map({0:'b',1:'r'}))
from sklearn import svm
clf = svm.SVC(kernel='linear')
clf.fit(ex6data1.iloc[:,:1], ex6data1.iloc[:,2]) 
ex6data2 = pd.read_csv("../input/ex6data2.csv",header=None)
ex6data2.head()
ex6data2.plot.scatter(x=0,y=1,c=ex6data2[2].map({0:'b',1:'r'}))
from sklearn import svm
clf = svm.SVC()
clf.fit(ex6data2.iloc[:,:1], ex6data2.iloc[:,2]) 
ex6data3 = pd.read_csv("../input/ex6data3.csv",header=None)
ex6data3.head()
ex6data3.plot.scatter(x=0,y=1,c=ex6data3[2].map({0:'b',1:'r'}))
from sklearn import svm
clf = svm.SVC()
clf.fit(ex6data3.iloc[:,:1], ex6data3.iloc[:,2]) 
