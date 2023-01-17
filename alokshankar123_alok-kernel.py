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

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn import svm

from sklearn.neighbors import KNeighborsClassifier

df_train = pd.read_csv("../input/train.csv")
df_train.head()
df_train.isnull().sum()
df_train.count()
df_train[df_train['Sex'].str.match("female")].count()
df_train[df_train['Sex'].str.match("male")].count()

df_train['Pclass'].value_counts()
sns.countplot(x='Survived', hue='Pclass', data=df_train)
sns.countplot(x='Survived', hue='Sex', data=df_train)
plt.figure(figsize=(8,6))

sns.boxplot(x='Pclass',y='Age',data=df_train)
def check_age(age):

    if pd.isnull(age):

        return int(df_train["Age"].mean())

    else:

        return age
df_train['Age'] = df_train['Age'].apply(check_age)
df_train['Age'].isnull().sum()
df_train.drop(["Cabin","Name"],inplace=True,axis=1)
df_train.dropna(inplace=True)
pd.get_dummies(df_train["Sex"]).head()
sex = pd.get_dummies(df_train["Sex"])
embarked = pd.get_dummies(df_train["Embarked"])

pclass = pd.get_dummies(df_train["Pclass"])
df_train = pd.concat([df_train,pclass,sex,embarked],axis=1)
df_train.head()
df_train.drop(["PassengerId","Pclass","Sex","Ticket","Embarked"],axis=1,inplace=True)
df_train.head()
X = df_train.drop("Survived",axis=1)

y = df_train["Survived"]
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)
#from sklearn.linear_model import LogisticRegression

lrmodel = LogisticRegression()

lrmodel.fit(X_train,y_train)

y_pred_lr = lrmodel.predict(X_test)



accuracy_score_lr = accuracy_score(y_pred_lr,y_test)

accuracy_score_lr

#predictions = logmodel.predict(X_test)

from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred_lr))
from sklearn.metrics import confusion_matrix

confusion_matrix(y_test, predictions)
# Decision Tree

dtree = DecisionTreeClassifier(criterion='entropy',max_depth = 4,random_state = 0)

dtree.fit(X_train,y_train)

y_pred_dtree = dtree.predict(X_test)
accuracy_score_dtree = accuracy_score(y_pred_dtree,y_test)

accuracy_score_dtree
# Random Forest

rf = RandomForestClassifier(criterion = 'gini',random_state = 0)

rf.fit(X_train,y_train)
y_pred_rf = rf.predict(X_test)

accuracy_score_rf = accuracy_score(y_pred_rf,y_test)

accuracy_score_rf
sv = svm.SVC(kernel= 'linear',gamma =2)

sv.fit(X_train,y_train)
#SVM

y_pred_svm = sv.predict(X_test)

accuracy_score_svm = accuracy_score(y_pred_svm,y_test)

accuracy_score_svm
#KNN

knn = KNeighborsClassifier(n_neighbors = 5)

knn.fit(X_train,y_train)
y_pred_knn = knn.predict(X_test)

accuracy_score_knn = accuracy_score(y_pred_knn,y_test)

accuracy_score_knn
scores = [accuracy_score_lr,accuracy_score_dtree,accuracy_score_rf,accuracy_score_svm,accuracy_score_knn]

scores = [i*100 for i in scores]

algorithm  = ['Logistic Regression','Decision Tree','Random Forest','SVM', 'K-Means']

index = np.arange(len(algorithm))

plt.bar(index, scores)

plt.xlabel('Algorithm', fontsize=10)

plt.ylabel('Accuracy Score', fontsize=5)

plt.xticks(index, algorithm, fontsize=10, rotation=30)

plt.title('Accuracy scores for each classification algorithm')

plt.ylim(80,100)

plt.show() 
feat_importances = pd.Series(rf.feature_importances_, index=X.columns)

feat_importances = feat_importances.nlargest(20)

feat_importances.plot(kind='barh')

plt.show()