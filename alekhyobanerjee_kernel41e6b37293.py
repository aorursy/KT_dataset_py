# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



import matplotlib.pyplot as plt

%matplotlib inline



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from sklearn.datasets import load_iris

iris=load_iris()
x=iris.data.tolist()

Y=iris.target
X=[]

for i in range(0,len(x)):

    X.append(x[i][::2])

X=np.array(X)
from sklearn.model_selection import train_test_split

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2)
from sklearn.tree import DecisionTreeClassifier

clf=DecisionTreeClassifier()

clf.fit(X_train,Y_train)
Y_pred=clf.predict(X_test)
from sklearn.metrics import accuracy_score

accuracy_score(Y_test,Y_pred)
a=np.arange(start=X_train[:,1].min()-1,stop=X_train[:,1].max()+1,step=0.01)

b=np.arange(start=X_train[:,0].min()-1,stop=X_train[:,0].max()+1,step=0.01)
XX,YY=np.meshgrid(a,b)
input_array=np.array([XX.ravel(),YY.ravel()]).T

labels=clf.predict(input_array)
plt.contourf(XX,YY,labels.reshape(XX.shape))
plt.contourf(XX,YY,labels.reshape(XX.shape))

plt.scatter(X_train[:,1],X_train[:,0],c='#000000')

plt.show()