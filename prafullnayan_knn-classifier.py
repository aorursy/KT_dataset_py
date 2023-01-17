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



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df=pd.read_csv("../input/knn-data1/KNN_Project_Data")
df.head()


import matplotlib.pyplot as plt

from matplotlib.ticker import NullFormatter

import matplotlib.ticker as ticker

from sklearn import preprocessing

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.neighbors import KNeighborsClassifier
df.columns
df['TARGET CLASS'].value_counts()
df.hist(column='XVPM',bins=30)

plt.tight_layout()
sns.pairplot(data=df,hue='TARGET CLASS')
X=df.drop('TARGET CLASS',axis=1)
y=df['TARGET CLASS']
X=StandardScaler().fit(X).transform(X.astype(float))
X
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)

print('size of train set',X_train.shape,y_train.shape)

print('size of test set',X_test.shape,y_test.shape)
k=4

model=KNeighborsClassifier(n_neighbors=k)
model.fit(X_train,y_train)

pred=model.predict(X_test)
pred
from sklearn.metrics import accuracy_score,confusion_matrix,f1_score,classification_report
print("out sample accuracy",accuracy_score(pred,y_test))

print("in sample accuracy",accuracy_score(y_train,model.predict(X_train)))
confusion_matrix(pred,y_test)
f1_score(pred,y_test)


k_val=40

mean_acc=np.zeros((k_val-1))

for n in range(1,k_val):

    model1=KNeighborsClassifier(n_neighbors=n)

    model1.fit(X_train,y_train)

    pred=model1.predict(X_test)

    mean_acc[n-1]=accuracy_score(pred,y_test)





mean_acc    
p=np.arange(1,k_val)

plt.style.use('ggplot')

with plt.style.context('dark_background'):

    plt.figure(figsize=(12,8))

    plt.plot(p,mean_acc,'r-o')

    plt.legend(('Accuracy ', '+/- 3xstd'))

    plt.ylabel('Accuracy ')

    plt.xlabel('Number of Nabors (K)')

    plt.tight_layout()

plt.show()
print("best accuracy is",mean_acc.max(),'for k value=',mean_acc.argmax())
model1=KNeighborsClassifier(n_neighbors=11)

model1.fit(X_train,y_train)

pred=model1.predict(X_test)

print(classification_report(pred,y_test))