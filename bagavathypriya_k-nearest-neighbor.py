
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
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
df=pd.read_csv("../input/companydata/companydata.txt",index_col=0)
df.head()
x=df.drop('TARGET CLASS',axis=1)
from sklearn.preprocessing import StandardScaler
scal=StandardScaler()
scal.fit(x)
scale=scal.transform(x)
x=pd.DataFrame(scale,columns=df.columns[:-1])
x.head()
sns.pairplot(df,hue='TARGET CLASS')
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,df['TARGET CLASS'],test_size=0.30)
from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=1)
knn.fit(xtrain,ytrain)
pred=knn.predict(xtest)
from sklearn.metrics import confusion_matrix,classification_report
print('Confusion Matrix')
print(confusion_matrix(ytest,pred))
print('Classification report')
print(classification_report(ytest,pred))
from sklearn.model_selection import cross_val_score
acc=[]
for i in range(1,40):
    knn=KNeighborsClassifier(n_neighbors=i)
    score=cross_val_score(knn,x,df['TARGET CLASS'],cv=10)
    acc.append(score.mean())
    
plt.figure(figsize=(10,6))
plt.plot(range(1,40),acc,linestyle='dashed',marker='o',markersize=10,markerfacecolor='red')
plt.xlabel('K-neighbor')
plt.ylabel('Accuracy rate')
err=[]
for i in range(1,40):
    knn=KNeighborsClassifier(n_neighbors=i)
    score=cross_val_score(knn,x,df['TARGET CLASS'],cv=10)
    err.append(1-score.mean())
    
plt.figure(figsize=(10,6))
plt.plot(range(1,40),err,linestyle='dashed',marker='o',markersize=10,markerfacecolor='red')
plt.xlabel('K-neighbor')
plt.ylabel('Error rate')
knn = KNeighborsClassifier(n_neighbors=1)

knn.fit(xtrain,ytrain)
pred = knn.predict(xtest)

print('WITH K=1')
print('\n')
print(confusion_matrix(ytest,pred))
print('\n')
print(classification_report(ytest,pred))
knn = KNeighborsClassifier(n_neighbors=23)

knn.fit(xtrain,ytrain)
pred = knn.predict(xtest)

print('WITH K=23')
print('\n')
print(confusion_matrix(ytest,pred))
print('\n')
print(classification_report(ytest,pred))
