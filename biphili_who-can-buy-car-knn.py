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
import numpy as np

import matplotlib.pyplot as plt 

import pandas as pd 

import seaborn as sns

import warnings

warnings.filterwarnings('ignore') 

plt.style.use('fivethirtyeight')
dataset=pd.read_csv('../input/Social_Network_Ads.csv')

dataset.head()
print('Rows     :',dataset.shape[0])

print('Columns  :',dataset.shape[1])

print('\nFeatures :\n     :',dataset.columns.tolist())

print('\nMissing values    :',dataset.isnull().values.sum())

print('\nUnique values :  \n',dataset.nunique())
dataset.describe().T
f,ax=plt.subplots(1,2,figsize=(18,8))

dataset['Gender'].value_counts().plot.pie(explode=[0,0.05],autopct='%1.1f%%',ax=ax[0],shadow=True)

ax[0].set_title('Purchase by Gender')

ax[0].set_ylabel('Count')

sns.countplot('Gender',data=dataset,ax=ax[1],order=dataset['Gender'].value_counts().index)

ax[1].set_title('Purchase by Gender')

plt.show()
f,ax=plt.subplots(1,2,figsize=(18,8))

dataset['Purchased'].value_counts().plot.pie(explode=[0,0.05],autopct='%1.1f%%',ax=ax[0],shadow=True)

ax[0].set_title('Purchase Distribution')

ax[0].set_ylabel('Count')

sns.countplot('Purchased',data=dataset,ax=ax[1],order=dataset['Purchased'].value_counts().index)

ax[1].set_title('Purchase Distribution')

plt.show()
fig=plt.gcf()

fig.set_size_inches(10,7)

fig=sns.distplot(dataset['EstimatedSalary'],kde=True,bins=50);
X=dataset.iloc[:,[2,3]].values

y=dataset.iloc[:,4].values
from sklearn.model_selection import train_test_split   #cross_validation doesnt work any more

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0) 

#X_train
from sklearn.preprocessing import StandardScaler 

sc_X=StandardScaler()

X_train=sc_X.fit_transform(X_train)

X_test=sc_X.fit_transform(X_test)

#X_train
from sklearn.neighbors import KNeighborsClassifier

classifier=KNeighborsClassifier(n_neighbors=1,metric='minkowski',p=2)

classifier.fit(X_train,y_train)
y_pred=classifier.predict(X_test)
#from sklearn.metrics import confusion_matrix  #Class has capital at the begining function starts with small letters 

from sklearn.metrics import confusion_matrix,classification_report,accuracy_score

cm=confusion_matrix(y_test,y_pred)

import seaborn as sns

import matplotlib.pyplot as plt

f, ax = plt.subplots(figsize =(5,5))

sns.heatmap(cm,annot = True,linewidths=0.5,linecolor="red",fmt = ".0f",ax=ax)

plt.title("Test for Test Dataset")

plt.xlabel("predicted y values")

plt.ylabel("real y values")

plt.show()
print(classification_report(y_test,y_pred))
print(accuracy_score(y_test,y_pred))
from matplotlib.colors import ListedColormap

X_set,y_set=X_train,y_train

X1,X2=np.meshgrid(np.arange(start=X_set[:,0].min()-1,stop=X_set[:,0].max()+1,step=0.01),

                 np.arange(start=X_set[:,1].min()-1,stop=X_set[:,1].max()+1,step=0.01))

plt.contourf(X1,X2,classifier.predict(np.array([X1.ravel(),X2.ravel()]).T).reshape(X1.shape),

            alpha=0.75,cmap=ListedColormap(('red','green')))

plt.xlim(X1.min(),X1.max())

plt.ylim(X2.min(),X2.max())

for i,j in enumerate(np.unique(y_set)):

    plt.scatter(X_set[y_set==j,0],X_set[y_set==j,1],

               c=ListedColormap(('red','green'))(i),label=j)

plt.title('K-NN (Training set)')

plt.xlabel('Age')

plt.ylabel('Estimated Salary')

plt.show()
from matplotlib.colors import ListedColormap

X_set,y_set=X_test,y_test

X1,X2=np.meshgrid(np.arange(start=X_set[:,0].min()-1,stop=X_set[:,0].max()+1,step=0.01),

                 np.arange(start=X_set[:,1].min()-1,stop=X_set[:,1].max()+1,step=0.01))

plt.contourf(X1,X2,classifier.predict(np.array([X1.ravel(),X2.ravel()]).T).reshape(X1.shape),

            alpha=0.75,cmap=ListedColormap(('red','green')))

plt.xlim(X1.min(),X1.max())

plt.ylim(X2.min(),X2.max())

for i,j in enumerate(np.unique(y_set)):

    plt.scatter(X_set[y_set==j,0],X_set[y_set==j,1],

               c=ListedColormap(('red','green'))(i),label=j)

plt.title('K-NN (Test set)')

plt.xlabel('Age')

plt.ylabel('Estimated Salary')

plt.legend()

plt.show()
error_rate=[]



for i in range(1,40):

    knn=KNeighborsClassifier(n_neighbors=i)

    knn.fit(X_train,y_train)

    pred_i=knn.predict(X_test)

    error_rate.append(np.mean(pred_i != y_test))
plt.figure(figsize=(10,6))

plt.plot(range(1,40),error_rate,color='blue',linestyle='dashed',marker='o',markerfacecolor='red',markersize=10)

plt.title('Error Rate Vs K Value')

plt.xlabel('K')

plt.ylabel('Error Rate')

plt.show()
from sklearn.neighbors import KNeighborsClassifier

classifier=KNeighborsClassifier(n_neighbors=5,metric='minkowski',p=2)

classifier.fit(X_train,y_train)
y_pred=classifier.predict(X_test)
#from sklearn.metrics import confusion_matrix  #Class has capital at the begining function starts with small letters 

from sklearn.metrics import confusion_matrix,classification_report,accuracy_score

cm=confusion_matrix(y_test,y_pred)

import seaborn as sns

import matplotlib.pyplot as plt

f, ax = plt.subplots(figsize =(5,5))

sns.heatmap(cm,annot = True,linewidths=0.5,linecolor="red",fmt = ".0f",ax=ax)

plt.title("Test for Test Dataset")

plt.xlabel("predicted y values")

plt.ylabel("real y values")

plt.show()
print(classification_report(y_test,y_pred))
print(accuracy_score(y_test,y_pred))