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
import scipy

import numpy

import matplotlib

import pandas

import sklearn



# Load libraries

from pandas import read_csv

from pandas.plotting import scatter_matrix

from matplotlib import pyplot

from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC



#load data

dfTrain = pd.read_csv('../input/san-francisco-crime-classification/train.csv')

#show what data looks like

dfTrain

#exploratory analysis

dfTrain.info
dfTest = pd.read_csv('../input/san-francisco-crime-classification/test.csv')

dfTest
dfTest.info
print(dfTrain.describe)
loc = dfTrain.groupby("PdDistrict").count()[[]]

loc = loc.sort_values(by='PdDistrict', ascending = False)

loc
cat = dfTrain.groupby("Category").count()

cat