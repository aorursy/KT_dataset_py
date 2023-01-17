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

import warnings

warnings.filterwarnings('ignore') 
dataset=pd.read_csv('../input/Wine.csv')

dataset.head()
X=dataset.iloc[:,0:13].values

y=dataset.iloc[:,13].values
from sklearn.model_selection import train_test_split   #cross_validation doesnt work any more

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0) 

#X_train
from sklearn.preprocessing import StandardScaler 

sc_X=StandardScaler()

X_train=sc_X.fit_transform(X_train)

X_test=sc_X.fit_transform(X_test)

#X_train
X_train.shape
from sklearn.decomposition import PCA

pca=PCA(n_components=None)

X_train=pca.fit_transform(X_train)

X_test=pca.transform(X_test)

explained_variance=pca.explained_variance_ratio_
explained_variance
from sklearn.decomposition import PCA

pca=PCA(n_components=2)

X_train=pca.fit_transform(X_train)

X_test=pca.transform(X_test)

explained_variance=pca.explained_variance_ratio_
pca.explained_variance_ratio_
np.identity(X.shape[1])
components=pca.transform(np.identity(X.shape[1]))
dataset.columns
pd.DataFrame(components,columns=['pc_1','pc_2'],index=["Alcohol","Malic_Acid","Ash","Ash_Alcanity","Magnesium","Total_Phenols","Flavanoids","Nonflavanoid_Phenols","Proanthocyanins","Color_Intensity","Hue","OD280","Proline"])
X_train.shape
from sklearn.linear_model import LogisticRegression

classifier=LogisticRegression(random_state=0)

classifier.fit(X_train,y_train)
y_pred=classifier.predict(X_test)
from sklearn.metrics import confusion_matrix  #Class has capital at the begining function starts with small letters 

cm=confusion_matrix(y_test,y_pred)

import seaborn as sns

import matplotlib.pyplot as plt

f, ax = plt.subplots(figsize =(5,5))

sns.heatmap(cm,annot = True,linewidths=0.5,linecolor="red",fmt = ".0f",ax=ax)

plt.title("Test for Test Dataset")

plt.xlabel("predicted y values")

plt.ylabel("real y values")

plt.show()
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score

print(classification_report(y_test,y_pred))
print(accuracy_score(y_test,y_pred))
from matplotlib.colors import ListedColormap

X_set,y_set=X_train,y_train

X1,X2=np.meshgrid(np.arange(start=X_set[:,0].min()-1,stop=X_set[:,0].max()+1,step=0.01),

                 np.arange(start=X_set[:,1].min()-1,stop=X_set[:,1].max()+1,step=0.01))

plt.contourf(X1,X2,classifier.predict(np.array([X1.ravel(),X2.ravel()]).T).reshape(X1.shape),

            alpha=0.75,cmap=ListedColormap(('red','green','blue')))

plt.xlim(X1.min(),X1.max())

plt.ylim(X2.min(),X2.max())

for i,j in enumerate(np.unique(y_set)):

    plt.scatter(X_set[y_set==j,0],X_set[y_set==j,1],

               c=ListedColormap(('red','green','blue'))(i),label=j)

plt.title('Logistic Regression (Training set)')

plt.xlabel('PC1')

plt.ylabel('PC2')

plt.legend()

plt.show()
from matplotlib.colors import ListedColormap

X_set,y_set=X_test,y_test

X1,X2=np.meshgrid(np.arange(start=X_set[:,0].min()-1,stop=X_set[:,0].max()+1,step=0.01),

                 np.arange(start=X_set[:,1].min()-1,stop=X_set[:,1].max()+1,step=0.01))

plt.contourf(X1,X2,classifier.predict(np.array([X1.ravel(),X2.ravel()]).T).reshape(X1.shape),

            alpha=0.75,cmap=ListedColormap(('red','green','blue')))

plt.xlim(X1.min(),X1.max())

plt.ylim(X2.min(),X2.max())

for i,j in enumerate(np.unique(y_set)):

    plt.scatter(X_set[y_set==j,0],X_set[y_set==j,1],

               c=ListedColormap(('red','green','blue'))(i),label=j)

plt.title('Logistic Regression (Training set)')

plt.xlabel('PC1')

plt.ylabel('PC2')

plt.legend()

plt.show()