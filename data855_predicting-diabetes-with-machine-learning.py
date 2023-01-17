# importing libraries

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns



import warnings

warnings.filterwarnings('ignore')
data = pd.read_csv('../input/diabetes-data/pima-data.csv')
data.shape
data.info()
data.head()
# Histogram

columns = list(data)[0:-1]

data[columns].hist(stacked = False, bins = 100, figsize = (15, 30), layout = (14,2))
sns.barplot(x= data.bmi.value_counts().index, y = data.bmi.value_counts().values)

plt.xlabel('BMI')

plt.ylabel(" COUNT")

plt.show()
sns.barplot(x= data.num_preg.value_counts().index, y = data.num_preg.value_counts().values)

plt.xlabel('num_preg')

plt.ylabel(" COUNT")

plt.show()
sns.countplot(data.diabetes)

plt.xlabel('diabetes')

plt.ylabel('Count')

plt.show()
# correalation matrix

corrmat = data.corr()

top_corr_features = corrmat.index

plt.figure(figsize= (15,15))

g = sns.heatmap(data[top_corr_features].corr(), annot= True, cmap = 'RdYlGn')
#changing boolean to number



diabetes_map = {True: 1, False: 0}

data['diabetes']= data['diabetes'].map(diabetes_map)
data.head()
data.info()
n_true= len(data.loc[data['diabetes']== True])

n_false= len(data.loc[data['diabetes']== False])



print('Number of True Cases: {0}({1:2.2f}%)'.format(n_true, (n_true/(n_true+n_false))*100))

print('Number of False Cases: {0}({1:2.2f}%)'.format(n_false, (n_false/(n_true+n_false))*100))
#Start Building the models

from sklearn.model_selection import train_test_split
#spliting train and test data

x= data.drop(['diabetes'], axis= 1)

y= data.diabetes.values



X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state= 0)
from sklearn.impute import SimpleImputer



fill_values = SimpleImputer(missing_values=0, strategy = 'mean')



X_train = fill_values.fit_transform(X_train)

X_test = fill_values.fit_transform(X_test)
# support vector



from sklearn.svm import SVC

svm = SVC(random_state= 1)

svm.fit(X_train, y_train)

print("SVC accuracy: {: .2f}%".format(svm.score(X_test, y_test)*100))
# Naive Bayes



from sklearn.naive_bayes import GaussianNB

nb= GaussianNB()

nb.fit(X_train, y_train)

print("NB accuracy: {: .2f}%".format(nb.score(X_test, y_test)*100))
#KNN model



from sklearn.neighbors import KNeighborsClassifier

knn= KNeighborsClassifier(n_neighbors = 3)

knn.fit(X_train, y_train)

print("KNN accuracy: {: .2f}%".format(knn.score(X_test, y_test)*100))
# Random Forest 



from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators = 1000, random_state = 1)

rf.fit(X_train, y_train)

print("Random Forest accuracy: {: .2f}%".format(rf.score(X_test, y_test)*100))
# XG boost



import xgboost

xg = xgboost.XGBClassifier()

xg.fit(X_train, y_train)

print(" XG boost: {: .2f}%".format(xg.score(X_test, y_test)*100))