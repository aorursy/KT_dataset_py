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
#load the  dataset
data=pd.read_csv('../input/data.csv')
print(data.head())


#drop Unnammed: 32 column
data=data.drop(['Unnamed: 32'],axis=1)
print(data.head())
#checking for missing value
data.isnull().sum()
#Here Diagnosis is the target columns that contains value 'M' for Malignant 'B' for Benign. lets replace this value by 0 and 1
data['diagnosis'].replace({'M':0,'B':1},inplace=True)
#data set after replacing value 
data.head()
#drop id columns
data.drop(['id'],axis=1,inplace=True)
#Feature and target separation
X=data.drop(['diagnosis'],axis=1)
X.head()
y=data['diagnosis']
print(y.head())
#converting pandas data frame to numpy array
X=X.as_matrix()
y=y.as_matrix()
print(X)
print(y)
#Prinical Component Analysis
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D,axes3d

#principal components analysis

pca=PCA(n_components=3)
pca_comp=pca.fit(X).transform(X)

figure=plt.figure()
ax=Axes3D(figure,elev=20,azim=200)

figure.set_size_inches(10,10)
for i,colors,marker in zip([0,1],['red','green'],['o','^']):
        ax.scatter(pca_comp[:,0][y==i],pca_comp[:,1][y==i],pca_comp[:,2][y==i],color=colors,marker=marker,s=60)
plt.title(' 3 Pca Components')
plt.legend(['Malignant','Benign'])
plt.show()
plt.figure()
for i,colors,marker in zip([0,1],['red','green'],['o','^']):
        plt.scatter(pca_comp[:,0][y==i],pca_comp[:,1][y==i],color=colors,marker=marker,s=60)
plt.title('2 pca components')
plt.legend(['Malignant','Benign'])
plt.show()
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

lr=LogisticRegression()
estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('lr', lr))
model = Pipeline(estimators)
# evaluate pipeline
seed = 7
kfold = KFold(n_splits=5, random_state=seed)
results = cross_val_score(model, X, y, cv=kfold)
print("Mean Accuarcy logistics regression :{}".format(results.mean()))
print(results)

#Support vector machine
estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('lsvc', SVC(kernel='linear')))
model = Pipeline(estimators)
# evaluate pipeline
seed = 7
kfold = KFold(n_splits=5, random_state=seed)
results = cross_val_score(model, X, y, cv=kfold)
print("Mean Accuarcy support vector machine :{}".format(results.mean()))
print(results)

import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=33,shuffle=True)
sc=StandardScaler()
X_train=sc.fit(X_train).transform(X_train)
X_test=sc.transform(X_test)
feature_columns=tf.contrib.learn.infer_real_valued_columns_from_input(X_train)

dnn_clf=tf.contrib.learn.DNNClassifier(hidden_units=[16,16],feature_columns=feature_columns,n_classes=2)
dnn_clf.fit(x=X_train,y=y_train,batch_size=50,steps=10000)

print(dnn_clf.evaluate(X_test,y_test))
score=a=dnn_clf.evaluate(X_test,y_test)
print(score)
pre=dnn_clf.predict(np.array([X_test[0]]))
next(pre)