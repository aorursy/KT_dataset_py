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
df=pd.read_csv('/kaggle/input/breast-cancer-csv/breastCancer.csv')
df.head(5)
df.shape
df.isnull().sum().sum()
import seaborn as sns
sns.heatmap(df.isnull())
df1=df.drop('id',axis=1)
df1
df1.info()
df2=df1.drop('bare_nucleoli',axis=1)
df2
set(df2['class'])
x=df2.drop('class',axis=1)
y=df2['class']
x.columns
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.20)
print('Shape of X_train',X_train.shape)
print('Shape of y_train',y_train.shape)
print('Shape of X_test',X_test.shape)
print('Shape of y_text',y_test.shape)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
f=1
p=0
ii=0
yy=[]
xx=[]
i=1
while i<=40:
    
    classifier = KNeighborsClassifier(n_neighbors=i)
    classifier.fit(X_train, y_train)
    y_pred=classifier.predict(X_test)
    m=accuracy_score(y_test,y_pred)
    yy.append(m*100)
    xx.append(i)
    if p<m:
        p=m
        ii=i
    i+=1
print("At k=",ii,"we get maximum accuracy of ",round(p*100,2),'%')

import matplotlib.pyplot as plt
plt.figure(figsize=(16,9))
plt.plot(xx,yy,color='red')
plt.xlabel('Value of K----->',fontsize=15)
plt.ylabel('Accuracy in % ---->',fontsize=15)
plt.scatter(xx,yy)
plt.title('Accuracy vs Number of clusters ',fontsize=20,color='green')
