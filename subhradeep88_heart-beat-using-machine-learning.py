# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import matplotlib.pyplot as plt

import os

print(os.listdir("../input"))

import seaborn as sns

from sklearn.metrics import confusion_matrix

import warnings

warnings.filterwarnings('ignore')



# Any results you write to the current directory are saved as output.
dataset = pd.read_csv('../input/heart.csv')
dataset.head()
dataset.info()
dataset.describe()
dataset.shape
sns.countplot(x='sex',data=dataset)
plt.figure(figsize=(8,6))

explode =[0.1,0]

labels='Male','Female'

plt.pie(dataset['sex'].value_counts(),explode=explode,autopct='%1.1f%%',labels=labels,shadow=True,startangle=140)
plt.figure(figsize=(10,6))

explode=[0.1,0,0,0]

labels='Pain-Type 0','Pain Type-1','Pain-Type2','Pain-Type3'

plt.pie(dataset['cp'].value_counts(),explode=explode,labels=labels,autopct='%1.1f%%',shadow=True,startangle=140)
sns.boxplot(dataset['trestbps'],orient='v',color='Magenta')
sns.boxplot(dataset['chol'],orient='v',color='Magenta')


#dataset.plot.scatter(x='age',y='trestbps')

plt.figure(figsize=(20,10))

sns.boxplot(x='age',y='trestbps',data=dataset)
plt.figure(figsize=(20,10))

sns.boxplot(x='age',y='thalach',data=dataset)
sns.set()

col=['age','trestbps','chol','thalach']

sns.pairplot(dataset[col])

plt.show()
plt.figure(figsize=(15,10))

sns.heatmap(dataset.corr(),annot=True,cmap='YlGnBu')
sex = pd.get_dummies(dataset['sex'],prefix='sex',drop_first=True)

fbs = pd.get_dummies(dataset['fbs'],prefix='fbs',drop_first=True)

restecg = pd.get_dummies(dataset['restecg'],prefix='restecg',drop_first=True)

exang = pd.get_dummies(dataset['exang'],prefix='exang',drop_first=True)

cp = pd.get_dummies(dataset['cp'],prefix='cp',drop_first=True)

slope = pd.get_dummies(dataset['slope'],prefix='slope',drop_first=True)

thal = pd.get_dummies(dataset['thal'],prefix='thal',drop_first=True)



dataset = pd.concat([dataset,sex,fbs,restecg,exang,cp,slope,thal],axis=1)







#Will do a quick check if it worked or not :P

dataset.head()

dataset = dataset.drop(columns=['sex','fbs','restecg','exang','cp','slope','thal'])

dataset.head()
X= dataset.drop('target',axis=1)

y = dataset['target'].values
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test  = sc.transform(X_test)
from sklearn.decomposition import PCA

pca = PCA(n_components=None,random_state=0)

X_train = pca.fit_transform(X_train)

X_test =pca.transform(X_test)



pca.explained_variance_ratio_

from sklearn.linear_model import LogisticRegression



lr = LogisticRegression()

lr.fit(X_train,y_train)

y_pred = lr.predict(X_test)

lr_score = lr.score(X_test,y_test)



from sklearn.svm import SVC

sv = SVC(kernel ='rbf',random_state=0)

sv.fit(X_train,y_train)

sv_pred = sv.predict(X_test)

sv_score = sv.score(X_test,y_test)
from sklearn.ensemble import RandomForestClassifier

rf_regressor = RandomForestClassifier(n_estimators = 1000, random_state = 0)

rf_regressor.fit(X_train, y_train)

rf_pred = rf_regressor.predict(X_test)

rf_score = rf_regressor.score(X_test,y_test)

from sklearn.neighbors import KNeighborsClassifier



knn = KNeighborsClassifier(n_neighbors=2)

knn.fit(X_train,y_train)

knn_score = knn.score(X_test,y_test)

from sklearn.naive_bayes import GaussianNB

nv = GaussianNB()

nv.fit(X_train,y_train)

nv_sc = nv.score(X_test,y_test)


print("Logistic Regression Model Score is ",round(lr_score*100))

print("SVC Model Score is ",round(sv_score*100))

#print("Decision tree  Regression Model Score is ",round(tr_regressor.score(X_test,y_test)*100))

print("Random Forest Regression Model Score is ",round(rf_score*100))



print("KNeighbors Classifiers Model score is",round(knn_score*100))

print("Naive Bayes model score is",round(nv_sc*100))





from sklearn.model_selection import cross_val_score

accuracies_lr = cross_val_score(estimator = lr,X = X_train,y = y_train,cv = 10)

accuracies_sv = cross_val_score(estimator = sv,X = X_train,y = y_train,cv = 10)

accuracies_rf = cross_val_score(estimator = rf_regressor,X = X_train,y = y_train,cv = 10)



accuracies_knn = cross_val_score(estimator = knn,X = X_train,y = y_train,cv = 10)

accuracies_nv = cross_val_score(estimator = nv,X = X_train,y = y_train,cv = 10)



print("Mean Accuracies based on cross val score for logistic regression",round(accuracies_lr.mean()*100))

print("Mean Accuracies based on cross val score for SVM ",round(accuracies_sv.mean()*100))

print("Mean Accuracies based on cross val score for Random Forest",round(accuracies_rf.mean()*100))



print("Mean Accuracies based on cross val score for KNN",round(accuracies_knn.mean()*100))

print("Mean Accuracies based on cross val score for Naive Bayes",round(accuracies_nv.mean()*100))



cm_lr = confusion_matrix(y_test,y_pred)

cm_lr

cm_rf = confusion_matrix(y_test,rf_pred)

cm_rf