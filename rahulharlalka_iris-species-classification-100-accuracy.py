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
import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
dataset=pd.read_csv("../input/iris/Iris.csv")
dataset.head()
dataset.head()
dataset['Species'].value_counts()
sns.countplot(x='Species',data=dataset)

plt.savefig('count sprecied in train data.png')
species={'Iris-setosa':1,'Iris-versicolor':2,'Iris-virginica':3}
dataset=dataset.replace({'Species':species})
dataset.head()
dataset.info()
dataset.describe()
dataset.corr()[['Species']].sort_values(by='Species',ascending=False)
dataset.drop('Id',axis=1,inplace=True)
dataset.head()
dataset.corr()[['Species']].sort_values(by='Species',ascending=False)
sns.regplot(x='PetalLengthCm',y='Species',data=dataset)

plt.savefig('dependance of petal length of iris flowers on prediction.png')
sns.regplot(x='PetalWidthCm',y='Species',data=dataset)

plt.savefig('dependance of petal width of iris flowers on prediction.png')
sns.regplot(x='SepalLengthCm',y='Species',data=dataset)

plt.savefig('dependance of sepal length of iris flowers on prediction.png')
sns.regplot(x='SepalWidthCm',y='Species',data=dataset)

plt.savefig('dependance of sepal width of iris flowers on prediction.png')
x_train=dataset.drop('Species',axis=1)

y_train=dataset[['Species']]
from sklearn.model_selection import train_test_split as tts

x_train,x_test,y_train,y_test=tts(x_train,y_train,test_size=0.2,random_state=18)

x_train.shape
x_test.shape
y_train.shape
y_test.shape
from sklearn.linear_model import LogisticRegression

clf=LogisticRegression(C=112,max_iter=50)
clf.fit(x_train,np.array(y_train).ravel())
y_predict=clf.predict(x_test)
from sklearn.metrics import confusion_matrix

confusion_matrix(y_test,y_predict)
from sklearn.metrics import accuracy_score

accuracy_score(y_test,y_predict)*100
y_predict=pd.DataFrame(y_predict)

sns.countplot(x=0,data=y_predict)

plt.savefig('y_predict data.png')