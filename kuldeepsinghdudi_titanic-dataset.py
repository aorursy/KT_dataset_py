# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
test = pd.read_csv('/kaggle/input/titanic/test.csv')
train = pd.read_csv('/kaggle/input/titanic/train.csv')
train.head()
train.info()
test.info()
train.columns
sns.barplot(x=train['Pclass'], y=train['Survived'])
sns.barplot(x=train['Sex'], y=train['Survived'])
survived = train[train['Survived'] == 1]
not_survived = train[train['Survived'] == 0]

print ("Survived: %i (%.1f%%)"%(len(survived), float(len(survived))/len(train)*100.0))
print ("Not Survived: %i (%.1f%%)"%(len(not_survived), float(len(not_survived))/len(train)*100.0))
print ("Total: %i"%len(train))
train.Pclass.value_counts()
pclass_survived=train.groupby('Pclass').Survived.value_counts()
pclass_survived
pclass_survived.unstack(level=0).plot(kind='bar', subplots=False)
pclass_average=train[['Pclass','Survived']].groupby(['Pclass'],as_index=False).mean()
pclass_average
pclass_average.plot(kind='bar',subplots=False)
sns.barplot(x='Pclass',y='Survived',data=train)
sns.barplot(x='Sex',y='Survived',data=train)
sex_average=train[['Sex','Survived']].groupby(['Sex']).mean()
sex_average
sns.factorplot('Sex', 'Survived', hue='Pclass', size=4, aspect=2, data=train)
sns.factorplot(x='Pclass', y='Survived', hue='Sex', col='Embarked', data=train)
train[['Embarked','Survived']].groupby(['Embarked']).mean()
sns.barplot(x='Embarked',y='Survived',data=train)
train[['Parch','Survived']].groupby(['Parch']).mean()
sns.barplot(x='SibSp', y='Survived', ci=None, data=train)
sns.heatmap(train.drop('PassengerId',axis=1).corr(), annot=True)
combine=[train,test]
for dataset in combine:
    dataset['Title'] = dataset['Name'].str.extract(' ([A-Za-z]+)\.')
train.head()
pd.crosstab(train['Title'],train['Sex'])
for dataset in combine:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col', \
 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Other')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
    
train[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()

title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Other": 5}
for dataset in combine:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)
train.head()
for dataset in combine:
    dataset['Sex'] = dataset['Sex'].map({'female':1,'male':2}).astype(int)
train.head()
for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].fillna('S')
train.head()
train = pd.get_dummies(data=train, columns=['Embarked'], drop_first=True)
test = pd.get_dummies(data=test, columns=['Embarked'], drop_first=True)
combine = [test,train]
train.head()
for dataset in combine:
    age_avg = dataset['Age'].mean()
    age_std = dataset['Age'].std()
    age_null_count = dataset['Age'].isnull().sum()
    
    age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)
    dataset['Age'][np.isnan(dataset['Age'])] = age_null_random_list
    dataset['Age'] = dataset['Age'].astype(int)
    
train['AgeBand'] = pd.cut(train['Age'], 5)

print (train[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean())

for dataset in combine:
    dataset.loc[dataset['Age'] <= 16, 'Age']=0
    dataset.loc[(dataset['Age']>16) & (dataset['Age']<=32),'Age']=1
    dataset.loc[(dataset['Age']>32) & (dataset['Age']<=48),'Age']=2
    dataset.loc[(dataset['Age']>48) & (dataset['Age']<=60),'Age']=3
    dataset.loc[dataset['Age'] >60 , 'Age']=4
train.head()
for dataset in combine:
    dataset['Fare']=dataset['Fare'].fillna(dataset['Fare'].median())
train['FareBand'] = pd.qcut(train['Fare'], 4)
print (train[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean())
for dataset in combine:
    dataset.loc[ dataset['Fare'] <= 8, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 8) & (dataset['Fare'] <= 13), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 13) & (dataset['Fare'] <= 30), 'Fare']   = 2
    dataset.loc[ dataset['Fare'] > 30, 'Fare'] = 3
    dataset['Fare'] = dataset['Fare'].astype(int)

train_family=pd.Series(train['SibSp'] + train['Parch'], name ='Family')
test_family=pd.Series(test['SibSp'] + test['Parch'], name ='Family')
train['Family']=train_family
test['Family']=test_family
sns.barplot(x=train['Family'],y=train['Survived'])
combine=[train,test]
train.head()
test.head()
train = train.drop(['Name', 'SibSp', 'Parch', 'Ticket', 'Cabin', 'Family','PassengerId', 'AgeBand', 'FareBand'], axis=1)
test=test.drop(['Name', 'SibSp', 'Parch', 'Ticket', 'Cabin', 'Family'], axis=1)
test.head()
x=train.drop('Survived',axis=1)
y=train['Survived']
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=0.2, random_state=42)
log_reg = LogisticRegression()
log_reg.fit(x_train, y_train)
preds = log_reg.predict(x_valid)
lr=accuracy_score(preds, y_valid)
print(lr)
gauss = GaussianNB()
gauss.fit(x_train, y_train)
preds = gauss.predict(x_valid)
NB=accuracy_score(preds, y_valid)
print(NB)
svc = SVC()
svc.fit(x_train, y_train)
preds = svc.predict(x_valid)
svc=accuracy_score(preds, y_valid)
print(svc)
perc = Perceptron()
perc.fit(x_train, y_train)
preds = perc.predict(x_valid)
perc=accuracy_score(preds, y_valid)
print(perc)
dtc = DecisionTreeClassifier()
dtc.fit(x_train, y_train)
preds = dtc.predict(x_valid)
dtc=accuracy_score(preds, y_valid)
print(dtc)
rfc = RandomForestClassifier()
rfc.fit(x_train, y_train)
preds = rfc.predict(x_valid)
rfc=accuracy_score(preds, y_valid)
print(rfc)
knn = KNeighborsClassifier()
knn.fit(x_train, y_train)
preds = knn.predict(x_valid)
knn=accuracy_score(preds, y_valid)
print(knn)
sgd = SGDClassifier()
sgd.fit(x_train, y_train)
preds = sgd.predict(x_valid)
sgd=accuracy_score(preds, y_valid)
print(sgd)
models = pd.Series(['LogisticRegression', 'GaussianNB', 'SVM', 'Perceptron',
                   'DecisionTree', 'RandomForest', 'KNN', 'SGDClassifier', 'GradientBoostingClassifier'])
accuracies = pd.Series([lr, NB, svc, perc, dtc, rfc, knn, sgd])
scores = pd.DataFrame({'Model':models, 'Accuracies':accuracies}).sort_values(['Accuracies'], ascending=False)
scores
from sklearn.model_selection import RandomizedSearchCV

criterion=['gini', 'entropy']
n_estimators = [100, 250, 500 ,1000]
min_samples_split = [2, 5, 10, 15, 100]
min_samples_leaf = [1, 2, 5, 10]
max_depth = [5,10,15,20]

params = {'n_estimators':n_estimators,
         'min_samples_split':min_samples_split,
         'min_samples_leaf':min_samples_leaf,
         'max_depth':max_depth,
         'criterion':criterion}


rfc = RandomForestClassifier()
grid_search = RandomizedSearchCV(estimator=rfc, param_distributions=params, scoring='accuracy', n_iter=10,
                                 cv=5, verbose=2, random_state=42, n_jobs=4)
grid_search.fit(x_train, y_train)
print(grid_search.best_score_)
print(grid_search.best_params_)
n_neighbors = [5,8,11,14]


params = {'n_neighbors':n_neighbors}


knn = KNeighborsClassifier()
grid_search = RandomizedSearchCV(estimator=knn, param_distributions=params, scoring='accuracy', n_iter=10,
                                 cv=5, verbose=2, random_state=42, n_jobs=4)
grid_search.fit(x_train, y_train)
print(grid_search.best_score_)
print(grid_search.best_params_)
from sklearn.ensemble import GradientBoostingClassifier
xgb = GradientBoostingClassifier()
xgb.fit(x_train, y_train)
preds = xgb.predict(x_valid)
xgb_acc = accuracy_score(preds, y_valid)
print(xgb_acc)

learning_rates = [0.001, 0.01, 0.1, 1]
n_estimators = [100, 250, 500 ,1000]
min_samples_split = [2, 5, 10, 15, 100]
min_samples_leaf = [1, 2, 5, 10]
max_depth = [5,10,15,20]

params = {'learning_rate':learning_rates,
         'n_estimators':n_estimators,
         'min_samples_split':min_samples_split,
         'min_samples_leaf':min_samples_leaf,
         'max_depth':max_depth}


gbc = GradientBoostingClassifier()
grid_search = RandomizedSearchCV(estimator=gbc, param_distributions=params, scoring='accuracy', n_iter=10,
                                 cv=5, verbose=2, random_state=42, n_jobs=4)
grid_search.fit(x_train, y_train)
print(grid_search.best_score_)
print(grid_search.best_params_)
best_gbc = GradientBoostingClassifier(n_estimators=500, min_samples_split=5,
                                 min_samples_leaf=10, max_depth=10, learning_rate=0.01)
best_rfc = RandomForestClassifier(n_estimators= 500, min_samples_split= 15, min_samples_leaf=1,
                                  max_depth= 5, criterion= 'gini')
best_knn = KNeighborsClassifier(n_neighbors=11)

models = [best_gbc, best_rfc, best_knn]
for model in models:
    model.fit(x_train, y_train)
    preds = model.predict(x_valid)
    print(f'Accuracy = {accuracy_score(y_valid, preds)}')
best_gbc.fit(x, y)
ids = test['PassengerId']
preds = best_gbc.predict(test.drop('PassengerId', axis=1))
output = pd.DataFrame({ 'PassengerId' : ids, 'Survived': preds })
output.to_csv('gender_submission.csv', index=False)
