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
# import libraries:
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# import dataset: 
df=pd.read_csv("../input/Social_Network_Ads.csv")
df.head()
df.describe()
df.info()
# dataset vizualizations:
sns.pairplot(df,hue='Gender')
sns.scatterplot(df['EstimatedSalary'],df['Age'],hue='Gender',data=df)
sns.distplot(df['EstimatedSalary'],bins=10,kde=False)
sns.lineplot(df['EstimatedSalary'],df['Age'],hue='Gender',data=df)
sns.boxenplot(df['EstimatedSalary'],df['Age'],hue='Gender',data=df)
sns.countplot(df['Age'],hue='Gender',data=df)
# spliting the datset in the form of dependent and independent form :
X=df.iloc[:,[2,3]].values
y=df.iloc[:,4].values
X
y
# spliting the datset into the from train and test form"
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0)
# feature scaling:
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.fit_transform(X_test)
# Apply the kernel PCA:
from sklearn.decomposition import KernelPCA
k_pca=KernelPCA(n_components=2,kernel='rbf')
X_train=k_pca.fit_transform(X_train)
X_test=k_pca.fit_transform(X_test)
# now the model is train on Logistic regression
from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression(random_state=0)
classifier.fit(X_train,y_train)
# prediction of new result:
y_pred=classifier.predict(X_test)
y_pred
# matric:
from sklearn.metrics import classification_report,confusion_matrix
cm=confusion_matrix(y_test,y_pred)
cm
cr=classification_report(y_test,y_pred)
cr
y_pred
y_test
