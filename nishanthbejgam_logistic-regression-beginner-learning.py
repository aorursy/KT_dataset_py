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
import matplotlib.pyplot as plt

import scipy.stats as st

import warnings

from sklearn.metrics import confusion_matrix

warnings.filterwarnings('ignore')
Dataset = pd.read_csv('../input/framingham_heart_disease.csv')
Dataset.keys()
Dataset.drop(['education'],axis=1,inplace=True)
Dataset.head()
Dataset.shape
Dataset.rename(columns={'male':'Sex_Male'},inplace=True)
Dataset.shape
Dataset.keys()
Dataset.isnull().sum()
Dataset.loc[Dataset['cigsPerDay'].isnull()==True].count()
count=0

for i in Dataset.isnull().sum(axis=1):

    if i>0:

        count+=1

count
Dataset.dropna(axis=0,inplace=True)
Dataset.describe()
Xloc=Dataset.iloc[:,0:14].values

Yloc=Dataset.iloc[:,14:15].values
from sklearn.model_selection import train_test_split

XTrain,XTest,YTrain,YTest = train_test_split(Xloc,Yloc,test_size=0.20,random_state=0)


from sklearn.linear_model import LogisticRegression

Classifier = LogisticRegression()

Classifier.fit(XTrain,YTrain)

YPred=Classifier.predict(XTest)

YPred.shape

YTest.shape
CM=confusion_matrix(YTest,YPred)
CM
from sklearn.metrics import accuracy_score
AC=accuracy_score(YTest,YPred)
AC