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
data = pd.read_csv('/kaggle/input/diamonds/diamonds.csv')
data.head(3)
#Let me drop the first column
del data['Unnamed: 0']
data.isnull().sum()
import seaborn as sns
import matplotlib.pyplot as plt
x= sns.PairGrid(data)
x = x.map(plt.scatter)
#Let us see the correlation between the columns
corr = data.corr()
sns.heatmap(corr, 
        xticklabels=corr.columns,
        yticklabels=corr.columns)
print(data.cut.unique())
print(data.color.unique())
print(data.clarity.unique())
data1 = data
from sklearn.preprocessing import LabelEncoder 
  
le = LabelEncoder() 
  
data1['cut_encoded']= le.fit_transform(data['cut']) 
data1['color_encoded']= le.fit_transform(data['color'])
data1['clarity_encoded']=le.fit_transform(data['clarity'])
data1.head()
data1.corr()
simple = data1[['carat','x','y','z','price']]
simple.head()
simple.shape
simple.corr()
X=simple.drop(columns=['price'])
y=simple.price

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)
#shape of training and test set

print(X_train.shape)
print(X_test.shape)
from sklearn.linear_model import *

lr = LinearRegression()

lr.fit(X_train,y_train)

print(lr.score(X_test,y_test))
