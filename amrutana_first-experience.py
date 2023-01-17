# importing required libraries

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.decomposition import PCA

from sklearn import cross_validation

from sklearn import preprocessing

from sklearn.cross_validation import cross_val_score

from sklearn.cross_validation import train_test_split

from sklearn.cross_validation import KFold

from sklearn import svm

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn import linear_model

from sklearn.ensemble import AdaBoostRegressor

from sklearn.ensemble import BaggingRegressor

from sklearn.ensemble import AdaBoostClassifier

from sklearn.ensemble import RandomForestRegressor

from sklearn.naive_bayes import GaussianNB

from sklearn import metrics

from sklearn.metrics import make_scorer
#Reading Data

df = pd.read_csv("../input/data.csv")

print (df.head())
# Dealing with missing data

total= df.isnull().sum().sort_values(ascending=False)

percent= df.isnull().sum()/df.isnull().count().sort_values(ascending=False)

missing_data= pd.concat([total,percent],axis=1,keys=['Total','Percent'])

print (missing_data.head(20))
#Dropping unwanted features

df= df.drop(['id'],axis=1)

df=df.drop("Unnamed: 32",axis=1)

#Convering Categorical Data

df['diagnosis']= pd.get_dummies(df['diagnosis'])
# PCA

df_1=df.iloc[:,1:]

pca=PCA(n_components=3)

pca.fit(df_1)



#Dimension indexing

dimensions= ['Dimentions {}'.format(i) for i in range(1,len(pca.components_)+1)]



#Individual PCA Components

components= pd.DataFrame(np.round(pca.components_,4),columns=df_1.keys())

components.index=dimensions



#Explained Variance in PCA

ratios= pca.explained_variance_ratio_.reshape(len(pca.components_),1)

variance_ratios= pd.DataFrame(np.round(ratios,4),columns= ['Explained Variance'])

variance_ratios.index=dimensions



print (pd.concat([variance_ratios,components],axis=1))
#Separating Malignant and Benign for graphs

malignant= df[df['diagnosis']==0]

benign= df[df['diagnosis']==1]
# Considering first 10 features. The features'area_worst' and 'perimeter_worst' is included as these features contribute to the Explained Variance.

observe = list(df.columns[1:11]) + ['area_worst'] + ['perimeter_worst']

df_1 = df.loc[:,observe]
plt.rcParams.update({'font.size':8})

plot, graphs= plt.subplots(6,2,figsize=(8,10))

graphs= graphs.flatten()

for idx,graph in enumerate(graphs):

    graph.figure

    

    binwidth= (max(df[observe[idx]])-min(df[observe[idx]]))/50

    bins = np.arange(min(df[observe[idx]]), max(df[observe[idx]]) + binwidth, binwidth)

    graph.hist([malignant[observe[idx]],benign[observe[idx]]], bins=bins, alpha=0.6, normed=True, label=['Malignant','Benign'], color=['red','blue'])

    graph.legend(loc='upper right')

    graph.set_title(observe[idx])

plt.tight_layout()

plt.show()
#Dropping the Unwanted variables

df_1 = df_1.drop(['symmetry_mean','fractal_dimension_mean','smoothness_mean','texture_mean'],axis=1)
#Splitting the Data

x = df_1

y = df['diagnosis']
#Random Forest 

Forest = RandomForestClassifier(n_estimators = 10)

Forest = Forest.fit(x,y)

Kfold = KFold(len(df_1),n_folds=10,shuffle=False)

print("KfoldCross Validation score using Random Forest is %s" %cross_val_score(Forest,x,y,cv=10).mean())



X_train,X_test,y_train,y_test = train_test_split(x,y,test_size= .2,random_state=0)

    

rf = Forest.fit(X_train,y_train)

y_pred = rf.predict(X_test)

print ("Accuracy: %s" %metrics.accuracy_score(y_test,y_pred))
#SVM

from sklearn import svm

svm_1 = svm.SVC(kernel='linear', C=1)

svm_1 = svm_1.fit(x,y)

Kfold = KFold(len(df_1),n_folds=10,shuffle=False)

print("KfoldCross Validation score using SVM is %s" %cross_val_score(svm_1,x,y,cv=10).mean())



X_train,X_test,y_train,y_test = train_test_split(x,y,test_size= .2,random_state=0)

    

svm_2 = svm_1.fit(X_train,y_train)

y_pred = svm_2.predict(X_test)

print ("Accuracy: %s" %metrics.accuracy_score(y_test,y_pred))
#Gaussian Naive Bayes

gnb= GaussianNB()

gnb = gnb.fit(x,y)

Kfold = KFold(len(df_1),n_folds=10,shuffle=False)

print("KfoldCross Validation score using Gaussian Naive Bayes is %s" %cross_val_score(gnb,x,y,cv=10).mean())



X_train,X_test,y_train,y_test = train_test_split(x,y,test_size= .2,random_state=0)

    

gnb_1 = gnb.fit(X_train,y_train)

y_pred = gnb_1.predict(X_test)

print ("Accuracy: %s" %metrics.accuracy_score(y_test,y_pred))
#Decision Tree

from sklearn import tree

Tree = tree.DecisionTreeClassifier()

Tree = Tree.fit(x,y)

Kfold = KFold(len(df_1),n_folds=10,shuffle=False)

print("KfoldCross Validation score using Decision Tree Classifier is %s" %cross_val_score(Tree,x,y,cv=10).mean())



X_train,X_test,y_train,y_test = train_test_split(x,y,test_size= .2,random_state=0)

    

Tree_1 = Tree.fit(X_train,y_train)

y_pred = Tree_1.predict(X_test)

print ("Accuracy: %s" %metrics.accuracy_score(y_test,y_pred))
#Logistic Regression

logistic = linear_model.LogisticRegression()

logistic = logistic.fit(x,y)

Kfold = KFold(len(df_1),n_folds=10,shuffle=False)

print("KfoldCross Validation score using Logistic Regression is %s" %cross_val_score(logistic,x,y,cv=10).mean())



X_train,X_test,y_train,y_test = train_test_split(x,y,test_size= .2,random_state=0)

    

logistic_1 = logistic.fit(X_train,y_train)

y_pred = logistic_1.predict(X_test)

print ("Accuracy: %s" %metrics.accuracy_score(y_test,y_pred))
#KNN

from sklearn import neighbors

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()

knn = knn.fit(x,y)

Kfold = KFold(len(df_1),n_folds=10,shuffle=False)

print("KfoldCross Validation score using KNN is %s" %cross_val_score(knn,x,y,cv=10).mean())



X_train,X_test,y_train,y_test = train_test_split(x,y,test_size= .2,random_state=0)

    

knn_1 = knn.fit(X_train,y_train)

y_pred = knn_1.predict(X_test)

print ("Accuracy: %s" %metrics.accuracy_score(y_test,y_pred))