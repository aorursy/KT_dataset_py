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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
data= pd.read_csv("../input/glass/glass.csv")
data.head()
data.shape
data['Type'].unique()
data.describe()
data.columns
#f,axes= plt.figure(1,2,figsize=14,12)

sns.countplot(data['Type'])
plt.xlabel("Type of glasses")
plt.ylabel("Count")
plt.show()
plt.scatter("Type","RI",data=data,c="Type")
plt.xlabel("Type of glasses")
plt.ylabel("RI")
plt.show()
#checking the distribution
cols= ['RI','Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe']
n_rows= 3
n_col=3 
fig,axs= plt.subplots(3,3,figsize=(18,12))

for r in range(0,n_rows):
    for c in range(0,n_col):
        i= r*n_col + c
        ax=axs[r][c]
        sns.distplot(data[cols[i]],ax=ax,kde=False)

help(sns.scatterplot)
#Realtionship b/w columns and type of glass
cols= ['RI','Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe', 'Type']
n_rows= 5
n_col= 2
fig,axs= plt.subplots(5,2,figsize=(18,14))

for r in range(0,n_rows):
    for c in range(0,n_col):
        i= r*n_col + c
        ax=axs[r][c]
        sns.scatterplot(data['Type'],data[cols[i]],ax=ax,hue=data['Type'],legend='full')
        plt.legend(loc='right')
        plt.tight_layout()
        i+=1

#sns.pairplot(data)
#plt.show()
data.isnull().sum()
#predicting the model
#spliting the data

X= data.iloc[:,:-1].values
Y= data.iloc[:,-1].values

from sklearn.model_selection import train_test_split

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.25,random_state=42)
#applying K-Nearest Neighbors (K-NN)

from sklearn.neighbors import KNeighborsClassifier
knn= KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train,Y_train)
y_pred= knn.predict(X_test)
print(y_pred)
from sklearn.metrics import accuracy_score
print("accuracy_score",accuracy_score(Y_test,y_pred))
a=[]
for i in range(1,15):
    knn= KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,Y_train)
    y_pred= knn.predict(X_test)
    #print(y_pred)
    print(f"accuracy_score when N_neighnors is {i}" ,accuracy_score(Y_test,y_pred)*100)
    
    a.append(accuracy_score(Y_test,y_pred)*100)
    i+=1
t= np.arange(1,15)
print(t)
a=np.abs(a)
print(a)
K_n= a[0]
sns.barplot(t,a,palette="vlag")
plt.show()
# Fitting classifier to the Training set
from sklearn.naive_bayes import GaussianNB
classifier_n = GaussianNB()
classifier_n.fit(X_train, Y_train)
y_pred1= classifier_n.predict(X_test)    
print("accuracy_score with Naive Bayes  is" ,accuracy_score(Y_test,y_pred1)*100)
GN=accuracy_score(Y_test,y_pred1)*100
#plot th graph b/w the best alog
modell=['Kneighobrs','Naive Bayes']
ac=[K_n,GN]

sns.barplot(modell,ac)
