import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score, precision_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
import seaborn as sns
import matplotlib.pyplot as plt
#load data
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
train.head()
train.describe()
train.info()
print('***************************')
test.info()
# Feature engineering regarding names.
words = ["Mr.", "Mrs.", "Miss.", "Capt.","Col.", "Major.", "Jonkheer.", "Don.", 'Sir.', "Dr.", "Rev.", "the Countess.", "Dona", "Mme", "Mlle", "Ms",  "Master", "Lady"]
train['Name'] = [' '.join(w for w in t.split() if w in words) for t in train['Name']]
test['Name'] = [' '.join(w for w in t.split() if w in words) for t in test['Name']]
train['Name'][:5]
Title_Dictionary = {
                        "Capt.":       "Officer",
                        "Col.":        "Officer",
                        "Major.":      "Officer",
                        "Jonkheer.":   "Royalty",
                        "Don.":        "Royalty",
                        "Sir." :       "Royalty",
                        "Dr.":         "Officer",
                        "Rev.":        "Officer",
                        "the Countess.":"Royalty",
                        "Dona.":       "Royalty",
                        "Mme.":        "Mrs",
                        "Mlle.":       "Miss",
                        "Ms.":         "Mrs",
                        "Mr." :        "Mr",
                        "Mrs." :       "Mrs",
                        "Miss." :      "Miss",
                        "Master." :    "Master",
                        "Lady." :      "Royalty"

                        }
newfeature= train['Name'].map(Title_Dictionary)
newfeature_test = test['Name'].map(Title_Dictionary)
newfeature.value_counts()
sns.barplot(x=newfeature, y ='Survived', data=train)
titles_dummy = pd.get_dummies(newfeature, prefix='Title')
train = pd.concat([train, titles_dummy], axis=1)
titles_dummy_test = pd.get_dummies(newfeature_test, prefix='Title')
test = pd.concat([test, titles_dummy_test], axis=1)
train.drop('Name',axis=1,inplace=True)
test.drop('Name',axis=1,inplace=True)
train.head()
# Sex (label encoding) and plot
train['Sex'] = train['Sex'].map({'female': 0, 'male': 1}).astype(int)
test['Sex'] = test['Sex'].map({'female': 0, 'male': 1}).astype(int)
test['Sex'].value_counts()
fig, axis1= plt.subplots(figsize=(8,3))
sns.countplot(x='Sex', data=train, ax=axis1)
sns.barplot(x='Sex', y ='Survived', data=train)
#Embarked

embark = train['Embarked'].fillna('S')
train['Embarked'] = embark.map({'S': 1, 'C': 2, 'Q': 3}).astype(int)
test['Embarked'] = embark.map({'S': 1, 'C': 2, 'Q': 3}).astype(int)
sns.barplot(x='Embarked', y ='Survived', data=train)
# Age
# handling with missing values
train['Age'] = train['Age'].fillna(train['Age'].median()).astype(int)
test['Age'] = test['Age'].fillna(train['Age'].median()).astype(int)
train['Age'].hist()
fig, axis1 = plt.subplots(1,1,figsize=(18,4))
der_data = train[['Age', 'Survived']].groupby(['Age'], as_index = False).mean()
sns.barplot(x='Age', y ='Survived', data = der_data)
Embarked_dummy = pd.get_dummies(train['Embarked'], prefix= 'Embarked')
train = pd.concat([train, Embarked_dummy], axis= 1)
Embarked_dummy_test = pd.get_dummies(test['Embarked'], prefix= 'Embarked')
test = pd.concat([test, Embarked_dummy_test], axis= 1)
train.drop('Embarked', axis=1,inplace=True)
test.drop('Embarked', axis=1,inplace=True)
train.head()
#lets replace Parch and SibSp with their sum
train['Relativesinship'] = train['SibSp'] + train['Parch']
test['Relativesinship'] = test['SibSp'] + test['Parch']
train[['Relativesinship', 'Survived']].groupby(['Relativesinship']).mean()
# only in test set fare feature has missing values
test['Fare'].fillna(test['Fare'].median(), inplace = True)
# converting floatings to integers
train['Fare'] = train['Fare'].astype(int)
test['Fare'] = test['Fare'].astype(int)
train.drop('Ticket',axis=1,inplace=True)
train.drop('Cabin',axis=1,inplace=True)
def age_cat(age):
    if age <= 16:
        return 0
    elif 16< age <=26:
        return 1
    elif 26< age <=36:
        return 2
    elif 36< age <=47:
        return 3
    elif 47 < age:
        return 4
    
train['Age'] = train['Age'].apply(age_cat)
test['Age'] = test['Age'].apply(age_cat)
Age_dummy = pd.get_dummies(train['Age'], prefix= 'Age')
train = pd.concat([train, Age_dummy], axis= 1)
Age_dummy_test = pd.get_dummies(test['Age'], prefix= 'Age')
test = pd.concat([test, Age_dummy_test], axis= 1)
train.drop('Age', axis=1, inplace=True)
test.drop('Age', axis=1, inplace=True)
train.head()
test.info()
# defining predictor and target values for machine learning models
X_train = train[['Pclass','Sex', 'Age_0','Age_1','Age_2', 'Age_3', 'Age_4', 'Relativesinship', 'Fare','Title_Miss', 'Title_Mr', 'Title_Mrs', 'Title_Officer', 'Embarked_1', 'Embarked_2', 'Embarked_3']]
y_train = train[['Survived']]
X_test = test[['Pclass','Sex', 'Age_0', 'Age_1', 'Age_2', 'Age_3', 'Age_4', 'Relativesinship', 'Fare','Title_Miss', 'Title_Mr', 'Title_Mrs', 'Title_Officer', 'Embarked_1', 'Embarked_2', 'Embarked_3']]

columns = ['Pclass','Sex', 'Age_0','Age_1','Age_2', 'Age_3', 'Age_4', 'Relativesinship', 'Fare','Title_Miss', 'Title_Mr', 'Title_Mrs', 'Title_Officer', 'Embarked_1', 'Embarked_2', 'Embarked_3']
X_train = X_train .reindex(columns= columns)
X_test = X_test.reindex(columns= columns)

X_train[columns] = X_train[columns].astype(int)
X_test[columns] = X_test[columns].astype(int)
X_train.info()
#Stochastic Gradient Descend classification
from sklearn.linear_model import SGDClassifier

sgd_clf = SGDClassifier(max_iter=5, random_state=42)
sgd_clf.fit(X_train, y_train)
cross_clf_score = cross_val_score(sgd_clf, X_train, y_train, cv = 10, scoring = 'accuracy')
cross_clf_score.mean()
# Confusion matrix
y_train_clf_pred = cross_val_predict(sgd_clf, X_train, y_train, cv=3)
confusion_matrix(y_train, y_train_clf_pred )
#precision/recall score
print(precision_score(y_train, y_train_clf_pred ))
print(recall_score(y_train, y_train_clf_pred ))
#Random Forest Classification
from sklearn.ensemble import RandomForestClassifier
forest_clf = RandomForestClassifier(random_state=42)
forest_clf.fit(X_train, y_train)
cross_forest_score = cross_val_score(forest_clf, X_train, y_train, cv = 10, scoring = 'accuracy')
cross_forest_score.mean()
#precision/recall score
y_train_pred_forest = cross_val_predict(forest_clf, X_train, y_train, cv = 3)
print('precision_score',precision_score(y_train, y_train_pred_forest))
print('recall_score',recall_score(y_train, y_train_pred_forest))
# K nearest classification
from sklearn.neighbors import KNeighborsClassifier
knn_clf = KNeighborsClassifier()
knn_clf.fit(X_train, y_train)
cross_knn_score = cross_val_score(knn_clf, X_train, y_train, cv = 10, scoring = 'accuracy')
cross_knn_score.mean()
#precision/recall score
y_train_pred_knn = cross_val_predict(knn_clf, X_train, y_train, cv = 3)
print('precision_score',precision_score(y_train, y_train_pred_knn))
print('recall_score',recall_score(y_train, y_train_pred_knn))
# Support vector machine classsification
from sklearn.svm import SVC
svc_clf = SVC()
svc_clf.fit(X_train, y_train)
cross_svc_score = cross_val_score(svc_clf, X_train, y_train, cv = 10, scoring = 'accuracy')
cross_svc_score.mean()
#precision/recall score
svc_clf_pred_train = cross_val_predict(svc_clf, X_train, y_train, cv = 3)
print('precision_score',precision_score(y_train, svc_clf_pred_train ))
print('recall_score',recall_score(y_train, svc_clf_pred_train ))
y_test_pred = svc_clf.predict(X_test)

svc_clf.score(X_train, y_train)
#Plot confusion matrix
conf_mx = confusion_matrix(y_train, svc_clf_pred_train )
plt.matshow(conf_mx, cmap=plt.cm.Blues)
mysubmission = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": y_test_pred
    })
mysubmission.to_csv('titanic.csv', index=False)