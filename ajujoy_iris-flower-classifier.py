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

dataset = pd.read_csv("../input/iris/Iris.csv") 
dataset.head(5)
dataset['Species'].value_counts()
import matplotlib.pyplot as plt 
dataset['Species'].value_counts().plot.pie(explode=[0.1,0.1,0.1],autopct='%1.1f%%',shadow=True,figsize=(10,8))
plt.show()


from  sklearn import  datasets
iris=datasets.load_iris()
x=iris.data
y=iris.target
iris
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.5)
x_train.shape,x_test.shape,y_train.shape,y_test.shape
from sklearn import tree
classifier=tree.DecisionTreeClassifier()
classifier
