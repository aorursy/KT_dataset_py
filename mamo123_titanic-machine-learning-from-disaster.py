import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

import seaborn as sns
test = pd.read_csv("../input/titanic/test.csv")

train = pd.read_csv("../input/titanic/train.csv")



y=train.Survived
train_features = train.drop(['Survived'], axis=1)

test_features = test

features = pd.concat([train_features, test_features]).reset_index(drop=True)
ax= sns.countplot(train['Survived'])

ax.set_title('No of Passengers survived')
ax=sns.countplot(x='Sex',hue ='Survived',data= train)

ax.set_title('Sex vs Survived')
ax=sns.countplot(x='Pclass',hue ='Survived',data= train)

ax.set_title('Pclass vs Survived')
ax=sns.countplot(x='Embarked',hue ='Survived',data= train)

ax.set_title('Embarked vs Survived')
ax=sns.violinplot("Sex","Age", hue="Survived", data=train,split=True)

ax.set_title('Sex and Age vs Survived')
ax=sns.violinplot("Pclass","Sex", hue="Survived", data=train, split=True)

ax.set_title('Pclass and Sex vs Survived')
ax=sns.countplot(x='SibSp',hue ='Survived',data= train)

ax.set_title('SibSp vs Survived')
sns.distplot(train[train['Pclass']==1].Fare)

#ax[0].set_title('Fares in Pclass 1')
g = sns.FacetGrid(train, col='Survived')

g = g.map(sns.distplot, "Age")
g = sns.heatmap(train[["Age","Fare","Sex","SibSp","Parch","Pclass","Survived"]].corr(),annot=True)
features= features.drop(['Name'],axis=1)

features= features.drop(['Ticket'],axis=1)

features= features.drop(['Cabin'],axis=1)
features.isna().sum()
from sklearn.impute import SimpleImputer

imp = SimpleImputer()

features.iloc[:, [3,6]] = imp.fit_transform(features.iloc[:,[3,6]].values)
features['Embarked'] = features['Embarked'].fillna('S')
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

features['Sex']=le.fit_transform(features['Sex'])

features['Embarked']=le.fit_transform(features['Embarked'])



from sklearn.preprocessing import OneHotEncoder

one = OneHotEncoder()

features = one.fit_transform(features).toarray()

features = pd.DataFrame(list(features))



from sklearn.preprocessing import StandardScaler

sd =StandardScaler()

features = sd.fit_transform(features)

features= pd.DataFrame(list(features))
train = features.iloc[:len(y),:].values

test = features.iloc[len(y):,:].values
from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.tree import DecisionTreeClassifier

from sklearn.svm import SVC
lin_reg = LogisticRegression()

lin_reg.fit(train,y)

lin_reg.score(train,y)



y_pred = lin_reg.predict(train)

test_y = lin_reg.predict(test)

lin_reg.score(test,test_y)
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y, y_pred)



from sklearn.metrics import precision_score, recall_score, f1_score

precision_score(y, y_pred)

recall_score(y, y_pred)

f1_score(y, y_pred)
knn = KNeighborsClassifier()

knn.fit(train, y)

y_pred = knn.predict(train)



knn.score(train,y)



y_test = knn.predict(test)
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y, y_pred)



from sklearn.metrics import precision_score, recall_score, f1_score

precision_score(y, y_pred)

recall_score(y, y_pred)

f1_score(y, y_pred)
nb = GaussianNB()

nb.fit(train, y)



y_pred = nb.predict(train)



nb.score(train,y)



y_test = nb.predict(test)



nb.score(test,y_test)
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y, y_pred)



from sklearn.metrics import precision_score, recall_score, f1_score

precision_score(y, y_pred)

recall_score(y, y_pred)

f1_score(y, y_pred)
svm = SVC()

svm.fit(train,y)

svm.score(train,y)

y_test = svm.predict(test)



svm.score(test,y_test)
dtf = DecisionTreeClassifier(criterion = 'entropy', max_depth = 11)

dtf.fit(train,y)

dtf.score(train,y)



y_pred = dtf.predict(test)

dtf.score(test,y_pred)