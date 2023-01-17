# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # using for plots
import seaborn as sns #using for plots
%matplotlib inline 

from sklearn.model_selection import train_test_split # split train and test sets
from sklearn.preprocessing import StandardScaler # for scaling 

from sklearn.tree import DecisionTreeClassifier 
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix

# Random Forest
from sklearn.ensemble import RandomForestClassifier
# Gradient Boosting Machine
from sklearn.ensemble import GradientBoostingClassifier
# Cross Validation Score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from time import time
from scipy.stats import randint as sp_randint
from sklearn.model_selection import RandomizedSearchCV

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import warnings
warnings.filterwarnings('ignore')

import os
print(os.listdir("../input"))


# Any results you write to the current directory are saved as output.
iris_ds= pd.read_csv("../input/Iris.csv")
iris_ds.columns,iris_ds.shape
#print(iris_ds.shape)
iris_ds.head()
iris_ds.tail()
iris_ds.isnull().sum()
iris_ds.dtypes
iris_ds.Species.value_counts()
iris_ds.drop(columns=['Id'],axis=1,inplace=True)

iris_ds.columns.values
iris_ds.hist(figsize=(20,10))
sns.pairplot(iris_ds,hue='Species')
sns.boxplot(data=iris_ds)
iris_ds.describe()
iris_ds.info()
iris_ds.Species = iris_ds.Species.astype('category')

iris_ds.info()
iris_ds.Species.cat.codes.head()
iris_ds.Species = iris_ds.Species.cat.codes
iris_ds.Species.tail()
iris_ds.columns.values
X = iris_ds[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
y = iris_ds.Species
x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
x_train.shape[0],y_train.shape[0]
scalar = StandardScaler()
x_train = scalar.fit_transform(x_train)
x_test = scalar.transform(x_test)
x_train
def normal_prediction():
    logis = LogisticRegression()
    logis.fit(x_train,y_train)
    print("logistic regression::\n",confusion_matrix(y_test,logis.predict(x_test)),"\n")
    
    svm = SVC()
    svm.fit(x_train,y_train)
    print("SVM ::\n",confusion_matrix(y_test,logis.predict(x_test)),"\n")
    
    knn = KNeighborsClassifier()
    knn.fit(x_train,y_train)
    print("KNN ::\n",confusion_matrix(y_test,knn.predict(x_test)),"\n")
    
    dTmodel = DecisionTreeClassifier()
    dTmodel.fit(x_train,y_train)
    print("DecisionTree ::\n",confusion_matrix(y_test,dTmodel.predict(x_test)),"\n")
    
    rForest = RandomForestClassifier()
    rForest.fit(x_train,y_train)
    print("RandomForest ::\n",confusion_matrix(y_test,rForest.predict(x_test)),"\n")

    grBoosting = GradientBoostingClassifier()
    grBoosting.fit(x_train,y_train)
    print("GradientBoosting ::\n",confusion_matrix(y_test,grBoosting.predict(x_test)),"\n")
normal_prediction()
#using cross_val_score
logis = LogisticRegression()
svm = SVC()
knn = KNeighborsClassifier()
dTmodel = DecisionTreeClassifier()
rForest = RandomForestClassifier()
grBoosting = GradientBoostingClassifier()
    
scores = cross_val_score(logis,x_train,y_train,cv=5)
print("Accuracy for logistic regresion: mean: {0:.2f} 2sd: {1:.2f}".format(scores.mean(),scores.std() * 2))
print("Scores::",scores)
print("\n")

scores2 = cross_val_score(svm,x_train,y_train,cv=5)
print("Accuracy for SVM: mean: {0:.2f} 2sd: {1:.2f}".format(scores2.mean(),scores2.std() * 2))
print("Scores::",scores)
print("\n")

scores3 = cross_val_score(knn,x_train,y_train,cv=5)
print("Accuracy for KNN: mean: {0:.2f} 2sd: {1:.2f}".format(scores3.mean(),scores3.std() * 2))
print("Scores::",scores)
print("\n")

scores4 = cross_val_score(dTmodel,x_train,y_train,cv=5)
print("Accuracy for Decision Tree: mean: {0:.2f} 2sd: {1:.2f}".format(scores4.mean(),scores4.std() * 2))
print("Scores::",scores4)
print("\n")

scores5 = cross_val_score(rForest,x_train,y_train,cv=5)
print("Accuracy for Random Forest: mean: {0:.2f} 2sd: {1:.2f}".format(scores5.mean(),scores5.std() * 2))
print("Scores::",scores5)
print("\n")

scores6 = cross_val_score(grBoosting,x_train,y_train,cv=5)
print("Accuracy for Gradient Boosting: mean: {0:.2f} 2sd: {1:.2f}".format(scores6.mean(),scores6.std() * 2))
print("Scores::",scores6)
print("\n")
clf = RandomForestClassifier()
#Random Forest
param_dist = {"max_depth": [3, None],
              "max_features": sp_randint(1, 4),
              "min_samples_split": sp_randint(2, 4),
              "bootstrap": [True, False],
              "criterion": ["gini", "entropy"]}

# run randomized search
n_iter_search = 5
random_search = RandomizedSearchCV(clf, param_distributions=param_dist,
                                   n_iter=n_iter_search, cv=5)

random_search.fit(x_train, y_train)
print(random_search.best_params_)
print(random_search.best_estimator_)
confusion_matrix(y_test,random_search.predict(x_test))

# use a full grid over all parameters
param_grid = {"max_depth": [3, None],
              "max_features": [1, 3, 4],
              "min_samples_split": [2, 3, 4],
              "bootstrap": [True, False],
              "criterion": ["gini", "entropy"]}

# run grid search
grid_search = GridSearchCV(clf, param_grid=param_grid, cv=5)

grid_search.fit(x_train, y_train)
print(grid_search.best_params_)
print(grid_search.best_estimator_)
confusion_matrix(y_test,grid_search.predict(x_test))
