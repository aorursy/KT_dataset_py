# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
df=pd.read_csv("../input/iris-flower-dataset/IRIS.csv")
df.head()
df.tail()
df.shape
df.describe()
df.info()
df['species'].value_counts()
df['species'].value_counts().plot(kind='bar')
plt.xlabel('Type')
plt.ylabel('Count')
sns.pairplot(df,hue='species')
corr=df.corr()
plot=sns.heatmap(corr,annot=True)
plt.show()
y=df['species']
x=df.drop("species",axis=1)
x.head()
y.unique()
from sklearn.preprocessing import LabelEncoder
lab=LabelEncoder()
y=lab.fit_transform(y)
y
from sklearn.preprocessing import StandardScaler
scale=StandardScaler()
scale.fit(x)
x=scale.transform(x)
x
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2,random_state=1)
from sklearn.neighbors import KNeighborsClassifier
model=KNeighborsClassifier(n_neighbors=3)
model.fit(xtrain,ytrain)
from sklearn.metrics import accuracy_score
ypredict=model.predict(xtest)
acc=accuracy_score(ypredict,ytest)*100
print(acc)
ypredict=model.predict([[4.6,3.1,1.5,0.2]])

if ypredict==0:
    print('Iris-setosa')
elif ypredict==1:
    print('Iris-versicolor')
elif ypredict==2:
    print('Iris-virginica')
