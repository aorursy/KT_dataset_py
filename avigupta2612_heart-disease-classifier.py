import numpy as np 

import pandas as pd 

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
df= pd.read_csv('../input/heart-disease-uci/heart.csv')

df.head()
#Check for missing values

df.isnull().sum()
df.describe()
sns.countplot(df['target'])
corrmap= df.corr()

plt.figure(figsize=(15,15))

sns.heatmap(corrmap, annot=True)
sns.countplot(df['sex'])
pd.crosstab(df.sex,df.target).plot(kind="bar",figsize=(10,4))

plt.title('Heart Disease Frequency for Sex')

plt.xlabel('Sex (0 = Female, 1 = Male)')

plt.xticks(rotation=0)

plt.legend(["Haven't Disease", "Have Disease"])

plt.ylabel('Frequency')

plt.show()

pd.crosstab(df.age, df.target).plot(kind='bar', figsize=(15,6))

plt.title('Heart Disease at different ages')

plt.xlabel('Age')

plt.ylabel('Frequency')

plt.show()
y= df['target']

train_data= df.drop('target', axis=1)

train_data.head()
#Normalize data

x= (train_data- np.min(train_data))/(np.max(train_data)-np.min(train_data)).values

x.head()
from sklearn.model_selection import train_test_split

train_x, test_x, train_y, test_y= train_test_split(x,y, test_size=0.2, random_state=4)

print(train_x.shape)

print(test_x.shape)
from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score
dtree= DecisionTreeClassifier()

dtree.fit(train_x, train_y)
dtree_pred= dtree.predict(test_x)

print('Decision Tree Accuracy: %f', accuracy_score(test_y, dtree_pred))
rforest= RandomForestClassifier()

rforest.fit(train_x, train_y)
forest_pred= rforest.predict(test_x)

print('Random Forest Accuracy: {}'.format(accuracy_score(test_y, forest_pred)))
LReg= LogisticRegression(C=1, class_weight='balanced',solver='liblinear')

LReg.fit(train_x,train_y)
Lreg_pred= LReg.predict(test_x)

print('Logistic Regression Accuracy: {}'.format(accuracy_score(test_y, Lreg_pred)))
Kneighbor= KNeighborsClassifier(n_neighbors=8)

Kneighbor.fit(train_x, train_y)
Kneighbor_pred= Kneighbor.predict(test_x)

print('K Neighbor Accuracy: {}'.format(accuracy_score(test_y, Kneighbor_pred)))
svm= SVC()

svm.fit(train_x, train_y)
svm_pred= svm.predict(test_x)

print('Support Vector Classifier Accuracy: {}'.format(accuracy_score(test_y, svm_pred)))