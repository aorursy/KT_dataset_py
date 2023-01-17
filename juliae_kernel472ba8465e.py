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
data=pd.read_csv('/kaggle/input/mushroom-classification/mushrooms.csv')
data.head(10)
data.isnull().sum()
from sklearn.preprocessing import LabelEncoder

labelencoder=LabelEncoder()

for col in data.columns:

    data[col] = labelencoder.fit_transform(data[col])

 

data.head()
import seaborn as sns

sns.set_style('whitegrid')

sns.countplot(x='class',hue='cap-shape',data=data,palette='rainbow')
data.corr()
DataX = data.iloc[:,1:23]  # all rows, all the features and no labels

DataY = data.iloc[:, 0]  # all rows, label only

DataX.head()

DataY.head()
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

DataX=scaler.fit_transform(DataX)

DataX
from sklearn.model_selection import train_test_split

trainX, testX, trainY, testY=train_test_split(DataX,DataY,test_size=0.2,random_state=4)
from sklearn.neighbors import KNeighborsClassifier

clf= KNeighborsClassifier()

clf.fit(trainX,trainY)

prediction=clf.predict(testX)

acc=clf.score(testX,testY)

print(acc)
from sklearn.metrics import confusion_matrix

confusion_matrix(testY,prediction)