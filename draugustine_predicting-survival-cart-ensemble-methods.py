import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

sns.set_style('whitegrid')

sns.set_context('poster')
train=pd.read_csv('../input/train.csv')

train.head()
sns.heatmap(train.isnull(), yticklabels=False, cbar=False, cmap='viridis')
sns.countplot(x='Survived', data=train)
sns.countplot(x='Survived', hue='Sex',  data=train, palette='RdBu_r')
sns.countplot(x='Survived', hue='Pclass',  data=train)
sns.distplot(train.Age.dropna(), kde=False, bins=30)
train.Age.plot.hist(bins=35)
train.info()
sns.countplot(x='SibSp', data=train)
train['Fare'].hist(bins=40, figsize=(10,4))

plt.xlabel('Price')

plt.ylabel('Count')
sns.boxplot(x='Pclass', y='Age', data=train)
def impute_age(cols):

    Age=cols[0]

    Pclass=cols[1]

    if pd.isnull(Age):

        if Pclass==1:

            return 37

        if Pclass==2:

            return 29

        if Pclass==3:

            return 24

    else: 

        return Age
train['Age']=train[['Age', 'Pclass']].apply(impute_age, axis=1)
sns.heatmap(train.isnull(), yticklabels=False, cbar=False, cmap='viridis')
train.head()
train=train.drop('Cabin', axis=1)
train.dropna()
sex=pd.get_dummies(train['Sex'], drop_first=True)
embarked=pd.get_dummies(train['Embarked'], drop_first=True)
train=pd.concat([train,sex, embarked], axis=1)
train.head()
train.drop(['Sex', 'Embarked', 'Name', 'Ticket'], axis=1, inplace=True)
train.head()
train.drop('PassengerId', axis=1, inplace=True)
train.head()
X=train.drop('Survived', axis=1)

y=train['Survived']
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.3, random_state=101)
from sklearn.linear_model import LogisticRegression
LR=LogisticRegression()
LR.fit(X_train, y_train)
y_pred=LR.predict(X_test)
from sklearn.metrics import classification_report, accuracy_score
print("Logistic Regression")

print(classification_report(y_test, y_pred))
print(accuracy_score(y_test, y_pred))
from sklearn.ensemble import RandomForestClassifier
RF=RandomForestClassifier(n_estimators=100, oob_score=True, random_state=42)
RF.fit(X_train, y_train)
y_pred_rf=np.around(RF.predict(X_test))
y_pred_rf
print("Random Forest Classifier")

print(classification_report(y_test, y_pred_rf))
print(accuracy_score(y_test, y_pred_rf))
feature_importance=pd.Series(RF.feature_importances_, index=X.columns)

feature_importance.sort_values().plot(kind='barh', color='g')
from sklearn.tree import DecisionTreeClassifier
DT=DecisionTreeClassifier()
DT.fit(X_train, y_train)
y_pred_DT=DT.predict(X_test)
print("Decision Tree Classifier")

print(classification_report(y_test, y_pred_DT))
print(accuracy_score(y_test, y_pred_DT))
from sklearn.ensemble import GradientBoostingClassifier
GBC=GradientBoostingClassifier(n_estimators=1000, learning_rate=1.0, max_depth=7, random_state=101)
GBC.fit(X_train, y_train)
y_pred_GBC=GBC.predict(X_test)
print("Gradient Boosting Classifier")

print(classification_report(y_test, y_pred_GBC))
print(accuracy_score(y_test, y_pred_GBC))
comparison=pd.DataFrame(np.array(['Logistic Regression', 'Random Forest Classifier', 'Decision Tree Classifier', 'Gradient Boosting']))
comparison.columns=['Method']
comparison['Precision']=[0.77, 0.81, 0.78, 0.81]

comparison['Recall']=[0.77, 0.81, 0.78, 0.81]

comparison['F1_Score']=[0.77, 0.81, 0.77, 0.80]

comparison['Accuracy']=[0.772, 0.813, 0.776, 0.806]

comparison
comparison.plot(kind='bar')

plt.ylim(0,1)

plt.xlim(-0.5, 7)

plt.xticks([0, 1, 2, 3], ['Logistic Regression', 'Random Forest Classifier', 'Decision Tree Classifier', 'Gradient Boosting Classifier'])
X_test_data=pd.read_csv('../input/test.csv')

X_test_data.head()
## missing data in test data

sns.heatmap(X_test_data.isnull(), yticklabels=False, cbar=False, cmap='viridis')
X_test_data['Age']=X_test_data[['Age', 'Pclass']].apply(impute_age, axis=1)
sex=pd.get_dummies(X_test_data['Sex'], drop_first=True)
embarked=pd.get_dummies(X_test_data['Embarked'], drop_first=True)
X_test_data=pd.concat([X_test_data,sex, embarked], axis=1)
X_test_data.drop(['Sex', 'Embarked', 'Name', 'Ticket'], axis=1, inplace=True)
PassengerId=X_test_data['PassengerId']
X_test_data.drop(['PassengerId', 'Cabin'], axis=1, inplace=True)
miss_fare=X_test_data[X_test_data['Pclass']==3].Fare.mean()
X_test_data.fillna(miss_fare, inplace=True)
y_predict_test=RF.predict(X_test_data)
Submission=pd.DataFrame(PassengerId, columns=['PassengerId'])

Submission['Survived']=y_predict_test
Submission.head()
Submission.to_csv('pred_test.csv', sep='\t')