import warnings

warnings.filterwarnings('ignore')



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline 
train_data = pd.read_csv("/kaggle/input/titanic/train.csv")

test_data = pd.read_csv("/kaggle/input/titanic/test.csv")
train_data.head(1)
test_data.head()
print("train_data_shape",train_data.shape)

print("test_data_shape",test_data.shape)
train_data.info()
test_data.info()
# train data

train_data['Pclass']=train_data['Pclass'].astype('object')

train_data['Survived']=train_data['Survived'].astype('object')

# test data

test_data['Pclass']=test_data['Pclass'].astype('object')
train_data.info()

print("@@@@@@")

test_data.info()
train_data.describe()
test_data.describe()
# percentage of missing values in train_data

train_data.isnull().sum()
test_data.isnull().sum()
plt.figure(figsize=(8,8))

sns.violinplot(x='Age',data=train_data)

plt.show()
train_data['Age'].fillna(train_data['Age'].median(), inplace=True)

test_data['Age'].fillna(test_data['Age'].median(), inplace=True)
pd.set_option('display.max_rows',891)

train_data['Cabin'].value_counts()
train_data['Cabin'] = train_data['Cabin'].replace(np.nan,'Q')

test_data['Cabin'] = test_data['Cabin'].replace(np.nan,'Q')
train_data['Embarked'].value_counts()
train_data.Embarked.mode()
train_data['Embarked'] = train_data['Embarked'].replace(np.nan,'S')
train_data.head(2)
train_data.isnull().sum()
train_data.head(3)
plt.figure(figsize = (10,10))

sns.heatmap(train_data.corr(),annot = True,cmap="tab20c")

plt.show()
test_data['Fare'].fillna(test_data['Fare'].median(),inplace=True)
test_data.isnull().sum()
plt.figure(figsize=(9,6))

sns.countplot('Survived',hue='Pclass',data=train_data,palette='twilight')
plt.figure(figsize=(9,6))

sns.countplot('Survived',hue='Sex',data=train_data,palette='twilight')
plt.figure(figsize=(8,5))

sns.barplot('Embarked','Survived',data=train_data,palette='muted')
plt.figure(figsize=(8,8))

sns.violinplot(y='Fare',x='Survived',hue='Survived',data=train_data)

plt.show()
plt.figure(figsize=(8,5))

sns.violinplot(x="Survived", y = "Age",data = train_data,palette='plasma',size=6)
train_data.Cabin[train_data.Survived==1].value_counts(ascending=False)
train_data.Cabin[train_data.Survived==0].value_counts(ascending=False)
train_data['Family'] = train_data['Parch']+train_data['SibSp']+1

test_data['Family'] = test_data['Parch']+test_data['SibSp']+1

# drop the Parch and SibSp columns

train_data = train_data.drop(['Parch','SibSp'],axis=1)

test_data = test_data.drop(['Parch','SibSp'],axis=1)
train_data.head(3)
plt.figure(figsize=(8,5))

sns.barplot('Family','Survived',data=train_data,palette='muted')
train_data = train_data.drop(['Name','Ticket'], axis = 1)

test_data = test_data.drop(['Name','Ticket'], axis = 1)
train_data['Cabin'].nunique()
train_data = train_data.drop(['Cabin'], axis = 1)

test_data = test_data.drop(['Cabin'], axis = 1)
train_data.info()
# Feature Scaling

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

train_data[['Age', 'Fare','Family']]= scaler.fit_transform(train_data[['Age', 'Fare','Family']])

train_data.head()
# Scaling on test data

test_data[['Age', 'Fare','Family']]= scaler.transform(test_data[['Age', 'Fare','Family']])

test_data.head()
# Creating a dummy variable for some of the categorical variables .

dummies = pd.get_dummies(train_data[['Pclass', 'Sex','Embarked']], drop_first=True)



# Adding the results to the master dataframe

train_data = pd.concat([train_data, dummies], axis=1)

train_data.head()
# Drop the existing columns

train_data=train_data.drop(['Pclass', 'Sex','Embarked'],axis=1)

train_data.head()
# One-Hot Encoding on test data

dummies = pd.get_dummies(test_data[['Pclass', 'Sex','Embarked']], drop_first=True)



# Adding the results to the master dataframe

test_data = pd.concat([test_data, dummies], axis=1)

test_data.head()
# Drop the existing columns

test_data=test_data.drop(['Pclass', 'Sex','Embarked'],axis=1)

test_data.head()
train_data.info()
train_data['Survived']=train_data['Survived'].astype('uint8')
# Spliting a dataset

X_train=train_data.drop(['Survived','PassengerId'], axis=1)

y_train=train_data["Survived"]
print("X_train shape", X_train.shape)

print("y_train shape", y_train.shape)
# drop uneccesary column from test data

test=test_data.drop(['PassengerId'], axis=1).copy()
print("test_data shape", test_data.shape)
# Logistic regression model

from sklearn.linear_model import LogisticRegression

L = LogisticRegression()
# Fitting the model on our trained dataset.

L.fit(X_train,y_train)
# Making Predictions

y_pred = L.predict(test)
# Calculating the accuracy of the model

print("Accuracy:",round(L.score(X_train, y_train)*100,2))
# printing the features coefficient

L.coef_
# List of features

features = ['Age', 'Fare','Family','Pclass_2','Pclass_3','Sex_male','Embarked_Q','Embarked_S']
# Listing the features in according to importance

coeff = pd.DataFrame(X_train.columns)

coeff.columns = ['Features']

coeff["Correlation"] = pd.Series(L.coef_[0])



coeff.sort_values(by='Correlation', ascending=False)
plt.figure(figsize=(8,5))

sns.barplot('Correlation','Features',data=coeff,palette='magma')
# Importing decision tree classifier from sklearn library

from sklearn.tree import DecisionTreeClassifier



# Fitting the decision tree with default hyperparameters, apart from

# max_depth which is 5 so that we can plot and read the tree.

dt_default = DecisionTreeClassifier(max_depth=5)

dt_default.fit(X_train, y_train)
# Making Predictions

y_pred = dt_default.predict(test)
# Calculating the accuracy

print("Accuracy:",round(dt_default.score(X_train, y_train)*100,2))
important_feature = pd.Series(dt_default.feature_importances_,index=features).sort_values(ascending=False)

important_feature
sns.barplot(important_feature.values,important_feature.index,palette='bone')
from sklearn.ensemble import RandomForestClassifier

RF= RandomForestClassifier(n_estimators=100,random_state=22)
# Fitting the model on our trained dataset.

RF.fit(X_train,y_train)
# Making Predictions

y_pred = RF.predict(test)
# Calculating the accuracy

print("Accuracy:",round(RF.score(X_train, y_train)*100,2))
feature_imp = pd.Series(RF.feature_importances_,index=features).sort_values(ascending=False)

feature_imp
sns.barplot(feature_imp.values,feature_imp.index,palette='bone')
submission = pd.DataFrame({

        "PassengerId": test_data["PassengerId"],

        "Survived": y_pred

    })

submission.to_csv('titanic_survival.csv', index=False)