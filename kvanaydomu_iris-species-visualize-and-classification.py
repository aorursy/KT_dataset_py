# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import numpy as np

from collections import Counter

from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix

from sklearn.pipeline import Pipeline

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import accuracy_score

import warnings

warnings.filterwarnings("ignore")



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
iris_data = pd.read_csv("/kaggle/input/iris/Iris.csv")
iris_data.head()
iris_data.shape
iris_data.info()
iris_data.columns
print("Our data have {} columns".format(len(iris_data.columns)))
withoutId=iris_data.drop("Id",axis=1)

withoutId.describe()
print("Numerical columns : ")

numerical_cols = [col for col in iris_data.columns if iris_data[col].dtype in ["int64","float64"]]

numerical_cols
print("Categorical columns ")

categorical_cols = [col for col in iris_data.columns if iris_data[col].dtype in ["object"]]

categorical_cols
iris_data['Species']
numerical_cols
uniqueTargets=iris_data.Species.unique()

uniqueTargets
iris_ver=iris_data[iris_data.Species == "Iris-versicolor"]

iris_set=iris_data[iris_data.Species == "Iris-setosa"]

iris_vir = iris_data[iris_data.Species == "Iris-virginica"]
iris_ver.drop("Species",axis=1)

iris_set.drop("Species",axis=1)

iris_vir.drop("Species",axis=1)
sns.heatmap(withoutId.corr(),annot=True,fmt="0.5f")

plt.title("Relationship between columns")

plt.show()
sns.distplot(iris_data.SepalLengthCm,kde=True)

plt.title("SepalLengthCm general distribition")

plt.show()
sns.kdeplot(iris_ver.SepalLengthCm,shade=True,label="Iris-versicolor")

sns.kdeplot(iris_set.SepalLengthCm,shade=True,label="Iris-setosa")

sns.kdeplot(iris_vir.SepalLengthCm,shade=True,label="Iris-virginica")

plt.title("SepalLengthCm distribiton according to Species")

plt.show()
sns.swarmplot(x=iris_data.Species,y=iris_data.SepalLengthCm)

plt.title("Compare SepalLengthCm values according to Species")

plt.show()
sns.distplot(a=iris_data.SepalWidthCm,kde=True)

plt.title("SepalWidthCm general distribition")

plt.show()
sns.kdeplot(iris_ver.SepalWidthCm,shade=True,label="Iris-versicolor")

sns.kdeplot(iris_set.SepalWidthCm,shade=True,label="Iris-setosa")

sns.kdeplot(iris_vir.SepalWidthCm,shade=True,label="Iris-virginica")

plt.title("SepalWidthCm distribiton according to Species")

plt.show()
sns.swarmplot(x=iris_data.Species,y=iris_data.SepalWidthCm)

plt.title("Compare SepalWidthCm values according to Species")

plt.show()

# It allows us see to distribitions 2 features according to constant(Species)
correlation=withoutId.corr()

print("Correlation between SepalWidthCm & SepalLengthCm is {}".format(correlation.SepalLengthCm.SepalWidthCm))
sns.scatterplot(x=iris_data.SepalWidthCm,y=iris_data.SepalLengthCm,hue=iris_data.Species)

plt.show("Obtain distribitons of SepalWidthCm & SepalLengthCm with scatter plot")

plt.show()
sns.lmplot(x="SepalWidthCm",y="SepalLengthCm",hue="Species",data=iris_data)

plt.title("Obtain distribitions of SepalWidthCm & SepalLengthCm with regression lines")

plt.show()
sns.kdeplot(iris_set.SepalWidthCm,shade=True,label="SepalWidthCm")

sns.kdeplot(iris_ver.SepalLengthCm,shade=True,label="SepalLengthCm")

plt.title("SepalWidthCm & SepalLengthCm distribition")

plt.show()
sns.distplot(a=iris_data.PetalLengthCm,kde=True)

plt.title("General PetalLengthCm distribition")

plt.show()
sns.kdeplot(iris_set.PetalLengthCm,shade=True,label="Iris-setosa")

sns.kdeplot(iris_ver.PetalLengthCm,shade=True,label="Iris-versicolor")

sns.kdeplot(iris_vir.PetalLengthCm,shade=True,label="Iris-virginica")

plt.title("PetalLengthCm distribition according to Speices")

plt.show()
sns.swarmplot(x=iris_data.Species,y=iris_data.PetalLengthCm)

plt.title("Compare PetalLengthCm values according to Species")

plt.show()
sns.distplot(a=iris_data.PetalLengthCm,kde=True)

plt.title("General PetalWidthCm distribition")

plt.show()
sns.kdeplot(iris_set.PetalWidthCm,shade=True,label="Iris-setosa")

sns.kdeplot(iris_ver.PetalWidthCm,shade=True,label="Iris-versicolor")

sns.kdeplot(iris_vir.PetalWidthCm,shade=True,label="Iris-virginica")

plt.title("PetalWidthCm distribition according to Speices")

plt.show()
sns.swarmplot(x=iris_data.Species,y=iris_data.PetalWidthCm)

plt.title("Compare PetalWidthCm values according to Species")

plt.show()
correlation=withoutId.corr()

print("Correlation between PetalWidthCm & PetalLengthCm is {}".format(correlation.PetalWidthCm.PetalLengthCm))
sns.kdeplot(iris_set.PetalWidthCm,shade=True,label="PetalWidthCm")

sns.kdeplot(iris_ver.PetalLengthCm,shade=True,label="PetalLengthCm")

plt.title("PetalWidthCm & PetalLengthCm distribition")

plt.show()
sns.jointplot(x=iris_data.PetalWidthCm,y=iris_data.PetalLengthCm,kind="kde")

plt.show()
sns.jointplot(x=iris_data.PetalWidthCm,y=iris_data.PetalLengthCm,color="red")

plt.show()
sns.scatterplot(x=iris_data.SepalWidthCm,y=iris_data.SepalLengthCm)

plt.show()
sns.scatterplot(x=iris_data.SepalWidthCm,y=iris_data.SepalLengthCm,hue=iris_data.Species)

plt.show()
sns.lmplot(x="PetalWidthCm",y="PetalLengthCm",hue="Species",data=iris_data)

plt.show()
print("Correlation between SepalLengthCm & PetalLengthCm is {}".format(correlation.SepalLengthCm.PetalLengthCm))
sns.kdeplot(iris_set.SepalLengthCm,shade=True,label="SepalLengthCm")

sns.kdeplot(iris_ver. PetalLengthCm,shade=True,label="PetalLengthCm")

plt.title("PetalWidthCm & PetalLengthCm distribition")

plt.show()
sns.jointplot(x=iris_data.SepalLengthCm,y=iris_data.PetalLengthCm,kind="kde")

plt.show()
sns.scatterplot(x=iris_data.SepalLengthCm,y=iris_data.PetalLengthCm)

plt.title("Distribion of SepalLengthCm & PetalLengthCm ")

plt.show()
sns.scatterplot(x=iris_data.SepalLengthCm,y=iris_data.PetalLengthCm,hue=iris_data.Species)

plt.title("Distribion of SepalLengthCm & PetalLengthCm according to Species")

plt.show()
sns.lmplot(x="SepalLengthCm",y="PetalLengthCm",hue="Species",data=iris_data)

plt.show()
print("Correlation between SepalLengthCm & PetalWidthCm is {}".format(correlation.SepalLengthCm.PetalWidthCm))
sns.kdeplot(iris_data.SepalLengthCm,label="SepalLengthCm",shade=True)

sns.kdeplot(iris_data.PetalWidthCm,label="PetalWidthCm",shade=True)

plt.title("Distribition of SepalLengthCm & PetalWidthCm")

plt.show()
sns.scatterplot(x=iris_data.PetalWidthCm,y=iris_data.SepalLengthCm)

plt.show()
sns.scatterplot(x=iris_data.PetalWidthCm,y=iris_data.SepalLengthCm,hue=iris_data.Species)

plt.title("Distribition of SepalLengthCm & PetalWidthCm with multiple colors in Scatter plot")

plt.show()
sns.lmplot(x="SepalLengthCm",y="PetalWidthCm",hue="Species",data=iris_data)

plt.show()
iris_data.isnull().sum()

# dataset has no missing values.Spectacular!
iris_data.dropna(axis=0,subset=["Species"],inplace=True)

y=iris_data.Species

X=iris_data.drop(["Id","Species"],axis=1)
grid = {"n_neighbors" : np.arange(1,50)}

knn = KNeighborsClassifier()

knn_cv = GridSearchCV(knn,grid,cv=3)

knn_cv.fit(X,y)

y_pred = knn_cv.predict(X)



print("Best value of K for KNN : {}".format(knn_cv.best_params_))

print("Best score : {}".format(knn_cv.best_score_))
n_neighbors = np.arange(1,31)

score = []



for i,k in enumerate(n_neighbors):

    knn=KNeighborsClassifier(n_neighbors=k)

    knn.fit(X,y)

    y_pred = knn.predict(X)

    score.append(accuracy_score(y,y_pred))



plt.plot(n_neighbors,score)

plt.xlabel("Values of k for KNN")

plt.ylabel("Scores")

plt.show()
y_pred = knn.predict(X)

y_true = y



cm1 = confusion_matrix(y_pred,y_true)

sns.heatmap(cm1,annot=True,fmt="0.5f")

plt.title("Table of errors belong to KNN(one dataset)")

plt.show()
grid = {"max_iter" : np.arange(1,300,10)}

logreg = LogisticRegression(random_state=42)

logreg_cv = GridSearchCV(logreg,grid,cv=3)

logreg_cv.fit(X,y)

y_pred = logreg_cv.predict(X)

print("Best value of max_iter for Logistic regression : {}".format(logreg_cv.best_params_))

print("Best score : {}".format(logreg_cv.best_score_))
y_pred = logreg_cv.predict(X)

y_true = y



cm2 = confusion_matrix(y_pred,y_true)

sns.heatmap(cm2,annot=True,fmt="0.5f")

plt.title("Table of errors belong to Logistic Regression(one dataset)")

plt.show()
grid = {"n_estimators" : np.arange(1,300,10)}

rfc = RandomForestClassifier(random_state=42)

rfc_cv = GridSearchCV(rfc,grid,cv=3)

rfc_cv.fit(X,y)

y_pred = rfc_cv.predict(X)

print("Best value of n_estimators for Random Forest Classifier : {}".format(rfc_cv.best_params_))

print("Best score : {}".format(rfc_cv.best_score_))
x_train,x_valid,y_train,y_valid = train_test_split(X,y,train_size=0.8,test_size=0.2,random_state=0)
print(x_train.shape)
print(x_valid.shape)
print(y_train.shape)
print(y_valid.shape)
grid = {"n_neighbors":np.arange(1,31)}

knn = KNeighborsClassifier()

knn_cv = GridSearchCV(knn,grid,cv=3)

knn_cv.fit(x_train,y_train)

y_pred = knn_cv.predict(x_valid)

print("Best value of k for KNN : {}".format(knn_cv.best_params_))

print("Best score : {}".format(knn_cv.best_score_))
n_neighbors = np.arange(1,31)

score = []



for i,k in enumerate(n_neighbors):

    knn=KNeighborsClassifier(n_neighbors=k)

    knn.fit(x_train,y_train)

    y_pred = knn.predict(x_valid)

    score.append(accuracy_score(y_valid,y_pred))



plt.plot(n_neighbors,score)

plt.xlabel("Values of k for KNN")

plt.ylabel("Scores")

plt.show()
knn = KNeighborsClassifier(n_neighbors=5)

knn.fit(x_train,y_train)

y1=knn.predict(x_valid)

print("KNN score ",accuracy_score(y1,y_valid))
clf = KNeighborsClassifier(n_neighbors=5)

scores_cv = cross_val_score(clf,x_train,y_train,cv=5)

print("Scores mean : {}".format(scores_cv.mean()))
y_pred_conf = knn.predict(x_valid)

y_true = y_valid

cm1 = confusion_matrix(y_pred,y_true)



plt.figure(figsize=(15,8))

sns.heatmap(cm1,annot=True,fmt="0.5f")

plt.title("Table of errors belong to KNN")

plt.show()
# C is logreg regularized parameter

grid = {"max_iter" : np.arange(1,300,10)}

param_grid = {'C': np.logspace(-3, 3, 7), 'penalty': ['l1', 'l2']}

logreg = LogisticRegression()

logreg_cv = GridSearchCV(logreg,param_grid,cv=3)

logreg_cv.fit(x_train,y_train)

print("Tuned hyperparamter n_neighbors : {}".format(logreg_cv.best_params_))

print("Best score : {}".format(logreg_cv.best_score_))
clf = LogisticRegression(max_iter=100,random_state=42)

scores_cv = cross_val_score(clf,x_train,y_train,cv=5)

print("Scores mean : {}".format(scores_cv.mean()))
y_pred_conf = logreg_cv.predict(x_valid)

y_true = y_valid

cm2 = confusion_matrix(y_pred,y_true)



plt.figure(figsize=(15,8))

sns.heatmap(cm2,annot=True,fmt="0.5f")

plt.title("Table of errors belong to Logistic Regression")

plt.show()
grid = {"n_estimators":np.arange(1,600,100)}

rfc = RandomForestClassifier(random_state=42)

rfc_cv = GridSearchCV(rfc,grid,cv=3)

rfc_cv.fit(x_train,y_train)

y_pred = rfc_cv.predict(x_valid)

print("Best value of n_estimators for Random Forest Classifier : {}".format(rfc_cv.best_params_))

print("Best score : {}".format(rfc_cv.best_score_))
clf = RandomForestClassifier(random_state=42,n_estimators=40)

scores_cv = cross_val_score(clf,x_train,y_train,cv=5)

print("Scores mean : {}".format(scores_cv.mean()))
y_pred_conf = rfc_cv.predict(x_valid)

y_true = y_valid

cm3 = confusion_matrix(y_pred,y_true)



plt.figure(figsize=(15,8))

sns.heatmap(cm3,annot=True,fmt="0.5f")

plt.title("Table of errors belong to Random Forest Classifier")

plt.show()
rfc = RandomForestClassifier(random_state=42,n_estimators=40)

rfc.fit(x_train,y_train)

rfc.predict([[3.2,4.7,1.8,2.9]])
iris_data.head(7)