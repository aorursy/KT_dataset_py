import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
iris  = sns.load_dataset("iris")
iris.head()
%matplotlib inline

sns.pairplot(iris,hue = 'species',height = 2)
X = iris.drop('species',axis = 1)
y = iris['species']
X.shape
y.shape
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,random_state = 1)
from sklearn.naive_bayes import GaussianNB
model  = GaussianNB()
model.fit(X_train,y_train)
y_model = model.predict(X_test)
model.score(X_test,y_test)
#alternate scoring
from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_model)
from sklearn.decomposition import PCA
model =  PCA(n_components = 2)
model.fit(X)
X_2d =model.transform(X)
iris['PCA1'] =X_2d[:,0]  #take all the data of the first dimension in the two-dimensional array
iris['PCA2'] =X_2d[:,1] #take all the data of the second dimension in the two-dimensional arra
sns.lmplot("PCA1","PCA2",hue ='species',data=iris,fit_reg = False)
#using Gaussian mixture model. It models the data as a collection of Gaussian blobs

from sklearn.mixture import GaussianMixture
model = GaussianMixture(n_components = 3)
model.fit(X)
y_gmm = model.predict(X)
iris['cluster']=y_gmm
sns.lmplot("PCA1","PCA2",data=iris,hue = 'species',col ='cluster',fit_reg = False)
