import numpy as np

import pandas as pd 

import seaborn as sns

import matplotlib.pyplot as plt

import os
dataset=pd.read_csv('/kaggle/input/mobile-price-classification/train.csv')

dataset.head()
dataset.info()
dataset.columns
dataset.describe()
corr=dataset.corr(method='pearson')

plt.figure(figsize=(19, 6))

sns.heatmap(corr,cmap="YlGnBu",annot=True)
sns.catplot(x='price_range',y='ram',data=dataset)
sns.catplot(x='price_range',y='int_memory',kind='swarm',data=dataset)
from matplotlib.pyplot import pie

values=dataset['three_g'].value_counts().values

pie(values,labels=['3G Supported','3G Not Supported'],autopct='%1.1f%%' ,shadow=True,startangle=90)
values=dataset['four_g'].value_counts().values

pie(values,labels=['4G Supported','4G Not Supported'],autopct='%1.1f%%' ,shadow=True,startangle=90)
sns.boxplot(x='price_range',y='battery_power',data=dataset)
X=dataset.iloc[:,:-1].values

y=dataset.iloc[:,-1].values
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 0)

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)

X_train

X_test
from sklearn.neighbors import KNeighborsClassifier

KNNclassifier=KNeighborsClassifier(n_neighbors=10)

KNNclassifier.fit(X_train,y_train)

y_pred = KNNclassifier.predict(X_test)

from sklearn.metrics import accuracy_score

accuracy_score(y_test,y_pred)*100
from sklearn.model_selection import cross_val_score

# Creating odd list K for KNN

neighbors = list(range(1,30))

# empty list that will hold cv scores

cv_scores = [ ]

#perform 10-fold cross-validation

for K in neighbors:

    knn = KNeighborsClassifier(n_neighbors = K)

    scores = cross_val_score(knn,X_train,y_train,cv = 10,scoring =

    "accuracy")

    cv_scores.append(scores.mean())
# Changing to mis classification error

mse = [1-x for x in cv_scores]

# determing best k

optimal_k = neighbors[mse.index(min(mse))]

print("The optimal no. of neighbors is {}".format(optimal_k))
mismatch=[]

for i in range(1,30):

    classifier=KNeighborsClassifier(n_neighbors=i)

    classifier.fit(X_train,y_train)

    y_pred=classifier.predict(X_test)

    mismatch.append(np.sum(y_pred != y_test))

    
plt.plot(range(1,30),mismatch)

plt.show()
from sklearn.neighbors import KNeighborsClassifier

KNNclassifier=KNeighborsClassifier(n_neighbors=22)

KNNclassifier.fit(X_train,y_train)

y_pred = KNNclassifier.predict(X_test)

from sklearn.metrics import accuracy_score

accuracy_score(y_test,y_pred)*100
from sklearn.svm import SVC

SVMlinear=SVC(kernel='linear')

SVMlinear.fit(X_train,y_train)

SVMlinear_predict=SVMlinear.predict(X_test)

y_pred = SVMlinear.predict(X_test)

accuracy_score(y_test,y_pred)*100

from sklearn.svm import SVC

SVMrbf=SVC(kernel='rbf')

SVMrbf.fit(X_train,y_train)

SVMrbf_predict=SVMrbf.predict(X_test)

y_pred = SVMrbf.predict(X_test)

accuracy_score(y_test,y_pred)*100
from sklearn.naive_bayes import GaussianNB

NB=GaussianNB()

NB.fit(X_train,y_train)

NB_predict=NB.predict(X_test)

y_pred = NB.predict(X_test)

accuracy_score(y_test,y_pred)*100
from sklearn.tree import DecisionTreeClassifier

DecisionTree=DecisionTreeClassifier(criterion='entropy',random_state=0)

DecisionTree.fit(X_train,y_train)

DecisionTree_predict=DecisionTree.predict(X_test)

y_pred = DecisionTree.predict(X_test)

accuracy_score(y_test,y_pred)*100
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import cross_val_score

trees = list(range(1,20))

# empty list that will hold cv scores

cv_scores = [ ]

#perform 10-fold cross-validation

for n in trees:

    RFC = RandomForestClassifier(n_estimators = n,criterion='entropy',random_state=0)

    scores = cross_val_score(RFC,X_train,y_train,cv = 10,scoring =

    "accuracy")

    cv_scores.append(scores.mean())
# Changing to mis classification error

mse = [1-x for x in cv_scores]

# determing best n

optimal_n = trees[mse.index(min(mse))]

print("The optimal no. of trees is {}".format(optimal_n))
from sklearn.ensemble import RandomForestClassifier

RFC=RandomForestClassifier(n_estimators=19,criterion='entropy',random_state=0)

RFC.fit(X_train,y_train)

RFC_predict=RFC.predict(X_test)

y_pred = RFC.predict(X_test)

accuracy_score(y_test,y_pred)*100