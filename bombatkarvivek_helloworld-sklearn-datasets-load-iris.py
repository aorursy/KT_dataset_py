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
from sklearn import datasets
iris = datasets.load_iris()

iris
digits = datasets.load_digits()

digits.data
from sklearn import svm
#classifier support vector machine 

clf = svm.SVC(gamma=0.001, C=100.)
clf.fit(digits.data[:-1],digits.target[:-1])
tmp = clf.predict(digits.data[-1:])

tmp