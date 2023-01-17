# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

#print(os.listdir("../input/Heart Disease UCI/heart.csv"))

df  = pd.read_csv("../input/heart.csv")

df.head()

df.shape

# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

sns.pairplot(df)
#set data and target from our dataset

data = df.iloc[:,0:-1]

target = df.iloc[:,13:14]

data.columns

target.columns
#now use sklearn for KNN 

from sklearn.neighbors import KNeighborsClassifier

reg = KNeighborsClassifier()

reg.fit(data,target)
#let us consider freature for target

reg.predict([[56,1,1,120,236,0,1,178,0,0.8,2,0,2]])