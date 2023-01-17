import warnings
warnings.filterwarnings('always')
warnings.filterwarnings('ignore')
import pandas_profiling

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

sns.set()
train = pd.read_csv('../input/titanic/train.csv')
test = pd.read_csv('../input/titanic/test.csv')
train.head()
test.head()
print("train:",train.shape)
print("test:",test.shape)
train.info()
print('_'*50)
test.info()
train.describe()
test.describe()
plt.figure(figsize=(10,5))
sns.heatmap(train.isnull(), cbar=False)
plt.figure(figsize=(10,5))
sns.heatmap(test.isnull(), cbar=False)
train.isnull().sum()
test.isnull().sum()
train['Age']= train['Age'].fillna(train['Age'].median())
test['Age']= test['Age'].fillna(test['Age'].median())
print(train['Age'].isnull().sum())
print(test['Age'].isnull().sum())
train["Cabin"].describe()
test["Cabin"].describe()
train['Cabin'][train['Cabin'].notnull()].head()
test['Cabin'][test['Cabin'].notnull()].head()
train["Cabin"] = pd.Series([i[0] if not pd.isnull(i) else 'M' for i in train['Cabin']])
test["Cabin"] = pd.Series([i[0] if not pd.isnull(i) else 'M' for i in test['Cabin']])
print(train["Cabin"].isnull().sum())
print(test["Cabin"].isnull().sum())
test['Cabin'].value_counts()
train['Embarked'].value_counts()
train['Embarked']= train['Embarked'].fillna(train['Embarked'].mode()[0])
train['Embarked'].isnull().sum()
test["Fare"] = test["Fare"].fillna(test["Fare"].median())
test.isnull().sum()
train.isnull().sum()
sns.pairplot(train)
plt.figure(figsize=(8,5))
sns.heatmap(train.corr(),annot=True,cmap='Purples')
plt.figure(figsize=(8,5))
sns.heatmap(test.corr(),annot=True,cmap='summer')
plt.figure(figsize=(8,5))
sns.countplot('Survived',hue='Pclass',data=train,palette='twilight')
plt.figure(figsize=(8,5))
sns.countplot('Sex',hue='Pclass',data=train,palette='mako')
plt.figure(figsize=(8,5))
sns.countplot('Embarked',hue='Pclass',data=train,palette='gist_heat')
train['Family_size']= train['SibSp']+train['Parch']+1
test['Family_size']= test['SibSp']+test['Parch']+1
plt.figure(figsize=(8,5))
sns.countplot('Family_size',hue='Pclass',data=train,palette='tab20')
plt.figure(figsize=(8,5))
train.groupby('Pclass')['Fare'].sum().plot(kind='bar')
plt.figure(figsize=(8,5))
sns.barplot('Cabin','Pclass',hue='Pclass',data=train,palette='Set1')
plt.figure(figsize=(8,5))
sns.boxplot(y="Age",x="Sex",hue="Pclass", data=train,palette='gray')
plt.figure(figsize=(8,5))
sns.countplot('Survived',hue='Sex',data=train,palette='summer')
plt.figure(figsize=(8,5))
sns.barplot(y="Survived", x="Sex", hue="Pclass", data=train, palette="ocean")
plt.figure(figsize=(8,5))
sns.violinplot(x="Survived", y = "Age",data = train,palette='plasma',size=6)
plt.figure(figsize=(8,5))
sns.boxplot(y="Age",x="Sex",hue="Pclass",data=train,palette='Purples')
plt.figure(figsize=(8,5))
sns.boxplot(y="Age",x="Family_size",hue="Survived", data=train,palette='rainbow')
plt.figure(figsize=(8,5))
sns.violinplot(y="Family_size",x='Survived',data = train,palette='hsv')
plt.figure(figsize=(8,5))
sns.barplot('Embarked','Survived',data=train,palette='muted')
plt.figure(figsize=(8,5))
sns.catplot("Pclass", col="Embarked",hue='Survived',data=train,kind='count',palette="inferno")
plt.figure(figsize=(8,5))
sns.catplot("Family_size", col="Embarked",hue='Survived',data=train,kind='count',palette="terrain")
plt.figure(figsize=(8,5))
sns.boxplot(y="Age",x="Embarked",hue='Survived',data=train, palette="cividis")
plt.figure(figsize=(8,5))
sns.distplot(train['Fare'],bins=30,color='m')
train["Fare"] = train["Fare"].map(lambda i: np.log(i) if i > 0 else 0)
test["Fare"] = test["Fare"].map(lambda i: np.log(i) if i > 0 else 0)
plt.figure(figsize=(8,5))
sns.distplot(train['Fare'],bins=30,color='m')
train.columns
codes = {"male": 0, "female": 1}
train['Sex']= train['Sex'].map(codes)
test['Sex']= test['Sex'].map(codes)
codes = {"S": 1, "C": 2, "Q": 3}
train['Embarked']= train['Embarked'].map(codes)
test['Embarked']= test['Embarked'].map(codes)
codes= {'M':1,'C':2,'B':3,'D':4,'E':5,'F':6,'A':7,'G':8,'T':9}
train['Cabin']= train['Cabin'].map(codes)
codes= {'M':1,'C':2,'B':3,'D':4,'E':5,'F':6,'A':7,'G':8}
test['Cabin']= test['Cabin'].map(codes)
train.head()
test.head()
train.drop(['SibSp','Parch','Ticket'],axis=1,inplace=True)
test.drop(['SibSp','Parch','Ticket'],axis=1,inplace=True)
train.drop('Name',axis=1,inplace=True)
test.drop('Name',axis=1,inplace=True)
train.head()
test.head()
X_train = train.drop(['Survived','PassengerId'], axis=1)
Y_train = train["Survived"]
X_test  = test.drop("PassengerId", axis=1).copy()
X_train.shape, Y_train.shape, X_test.shape
from sklearn.linear_model import LogisticRegression
LR = LogisticRegression()
# Fitting the model on our trained dataset.
LR.fit(X_train,Y_train)
# Making Predictions
y_pred = LR.predict(X_test)
# Calculating the Accuracy of the model.

print("Accuracy:",round(LR.score(X_train, Y_train)*100,2))
LR.coef_
features = ['Pclass', 'Sex', 'Age', 'Fare', 'Cabin', 'Embarked', 'Family_size']
coeff = pd.DataFrame(X_train.columns)
coeff.columns = ['Feature']
coeff["Correlation"] = pd.Series(LR.coef_[0])

coeff.sort_values(by='Correlation', ascending=False)
plt.figure(figsize=(8,5))
sns.barplot('Correlation','Feature',data=coeff,palette='magma')
from sklearn.ensemble import RandomForestClassifier
RF= RandomForestClassifier(n_estimators=100,random_state=22)
# Fitting the model on our trained dataset.
RF.fit(X_train,Y_train)
# Making Predictions
y_pred = RF.predict(X_test)
# Calculating the accuracy
print("Accuracy:",round(RF.score(X_train, Y_train)*100,2))
feature_imp = pd.Series(RF.feature_importances_,index=features).sort_values(ascending=False)
feature_imp
sns.barplot(feature_imp.values,feature_imp.index,palette='bone')
submission = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": y_pred
    })
submission.to_csv('titanic_2218.csv', index=False)