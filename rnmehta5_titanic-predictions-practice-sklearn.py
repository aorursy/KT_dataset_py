#importing required libraries
import numpy as np
import pandas as pd
from sklearn import tree, linear_model, metrics
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')
sns.set_style('whitegrid')

%matplotlib inline
#read the train and test sets and storing them in pd dataframes
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
print(train.shape)
print(test.shape)
train.head()
test.head()
for col in train.columns:
    print('number of null values in ' + col + ': ' + str(train[pd.isnull(train[col])].shape[0]))
for col in test.columns:
    print('number of null values in ' + col + ': ' + str(test[pd.isnull(test[col])].shape[0]))
train.pivot_table(index='Sex', values='Survived', aggfunc='mean').plot(kind='bar')
print(train.pivot_table(index='Sex', values='Survived', aggfunc='mean'))
fig, ax = plt.subplots(nrows=1, ncols=2, figsize = (12,5))
train.pivot_table(index='Sex', values='Survived', aggfunc='count').plot(kind='bar', ax=ax[0])
test.pivot_table(index='Sex', values='PassengerId', aggfunc='count').plot(kind='bar', ax=ax[1])
train.pivot_table(columns='Sex', index='Pclass',\
                  values='Survived', aggfunc='mean').plot(kind='bar')
print(train.pivot_table(columns='Sex', index='Pclass',\
                  values='Survived', aggfunc='mean'))
print(train.pivot_table(columns='Sex', index='Pclass',\
                  values='Survived', aggfunc='count'))
fig, ax = plt.subplots(nrows=1, ncols=2, figsize = (12,5))
train.pivot_table(index='Pclass', values='Survived', aggfunc='count').plot(kind='bar', ax=ax[0])
test.pivot_table(index='Pclass', values='PassengerId', aggfunc='count').plot(kind='bar', ax=ax[1])
train = train.join(pd.get_dummies(train.Sex))
test = test.join(pd.get_dummies(test.Sex))
class_dummies = pd.get_dummies(train.Pclass)
class_dummies.columns = ['Higher', 'Middle', 'Lower']
train = train.join(class_dummies)
X_train = train[['male', 'female','Higher', 'Middle', 'Lower']]
y = train['Survived']

class_dummies = pd.get_dummies(test.Pclass)
class_dummies.columns = ['Higher', 'Middle', 'Lower']
test = test.join(class_dummies)
X_test = test[['male', 'female', 'Higher', 'Middle', 'Lower']]
log_reg = linear_model.LogisticRegression()
baseline_log_reg = log_reg.fit(X_train, y)
predicted_survivors = baseline_log_reg.predict(X_test)

print('Accuracy: ', metrics.accuracy_score(train.Survived, baseline_log_reg.predict(X_train)))
sns.heatmap(metrics.confusion_matrix(train.Survived, baseline_log_reg.predict(X_train)),\
            cmap="Blues", annot=True)
plt.xlabel('Pred Label')
plt.ylabel('True Label')

test_baseline_model = test
test_baseline_model['Survived'] = predicted_survivors
test_baseline_model = test_baseline_model[['PassengerId', 'Survived']]

#test_baseline_model.to_csv('test_baseline_model.csv', index=False)
first_tree = tree.DecisionTreeClassifier()
#test['sex_mapped'] = train.Sex.map({'male':1, 'female':0})
first_tree_fit = first_tree.fit(X_train, y)
predicted_survivors = first_tree_fit.predict(X_test)

print('Accuracy: ', metrics.accuracy_score(train.Survived, first_tree_fit.predict(X_train)))
sns.heatmap(metrics.confusion_matrix(train.Survived, first_tree_fit.predict(X_train)), \
           cmap="Blues", annot=True)
plt.xlabel('Pred Label')
plt.ylabel('True Label')

test_first_model = test
test_first_model['Survived'] = predicted_survivors
test_first_model = test_first_model[['PassengerId', 'Survived']]

#test_first_model.to_csv('test_first_model.csv', index=False)
train.pivot_table(columns='Sex', index='Parch',\
                  values='Survived', aggfunc='mean').plot(kind='bar')
print(train.pivot_table(columns='Sex', index='Parch',\
                  values='Survived', aggfunc='mean'))
print(train.pivot_table(columns='Sex', index='Parch',\
                  values='Survived', aggfunc='count'))
train['new_Parch'] = train.Parch
train['new_Parch'] = train.new_Parch.astype(int)
train.loc[train.new_Parch > 1, 'new_Parch'] = 2
print(train.pivot_table(index=['Sex', 'Pclass'], columns='new_Parch', values='Survived', aggfunc='mean'))
print(train.pivot_table(index=['Sex', 'Pclass'], columns='new_Parch', values='Survived', aggfunc='count'))
train.pivot_table(index=['Sex', 'Pclass'], columns='new_Parch', values='Survived', aggfunc='mean').plot(kind='bar')
print(train.pivot_table(index='Sex', columns='SibSp', values='Survived', aggfunc='mean'))
print(train.pivot_table(index='Sex', columns='SibSp', values='Survived', aggfunc='count'))
train.pivot_table(index='Sex', columns='SibSp', values='Survived', aggfunc='mean').plot(kind='bar')
train['new_SibSp'] = train.SibSp
train['new_SibSp'] = train.SibSp.astype(int)
train.loc[train.new_SibSp > 1, 'new_SibSp'] = 2
print(train.pivot_table(index='Sex', columns='new_SibSp', values='Survived', aggfunc='mean'))
print(train.pivot_table(index='Sex', columns='new_SibSp', values='Survived', aggfunc='count'))
train.pivot_table(index='Sex', columns='new_SibSp', values='Survived', aggfunc='mean').plot(kind='bar')
X_train = train[['male', 'female', 'Pclass', 'new_Parch', 'new_SibSp']]
y = train['Survived']

test['new_SibSp'] = test.SibSp
test['new_SibSp'] = test.SibSp.astype(int)
test.loc[train.new_SibSp > 1, 'new_SibSp'] = 2

test['new_Parch'] = test.Parch
test['new_Parch'] = test.new_Parch.astype(int)
test.loc[train.new_Parch > 1, 'new_Parch'] = 2

X_test = test[['male', 'female', 'Pclass', 'new_Parch', 'new_SibSp']]
second_tree = tree.DecisionTreeClassifier()
#test['sex_mapped'] = train.Sex.map({'male':1, 'female':0})
second_tree_fit = second_tree.fit(X_train, y)
predicted_survivors = second_tree_fit.predict(X_test)

print('Accuracy: ', metrics.accuracy_score(train.Survived, second_tree_fit.predict(X_train)))
sns.heatmap(metrics.confusion_matrix(train.Survived, second_tree_fit.predict(X_train)), \
           cmap="Blues", annot=True)
plt.xlabel('Pred Label')
plt.ylabel('True Label')

test_second_model = test
test_second_model['Survived'] = predicted_survivors
test_second_model = test_second_model[['PassengerId', 'Survived']]

#test_second_model.to_csv('test_second_model.csv', index=False)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process.kernels import RBF
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score

model_names = [
    'KNeighborsClassifier',
    'SVC(kernel="linear")',
    'RandomForestClassifier',
    'AdaBoostClassifier()',
    'GaussianNB()',
]

models = [
    KNeighborsClassifier(),
    SVC(kernel="linear"),
    RandomForestClassifier(),
    AdaBoostClassifier(),
    GaussianNB()
]

fit_models = []

for i in range(len(models)):
    temp_score = cross_val_score(models[i], X_train, y, cv=5)
    fit_models.append(models[i].fit(X_train,y))
    print(model_names[i],' ', temp_score.mean())
    print('+/- ',temp_score.var())
test_rf_model = test
test_rf_model['Survived'] = fit_models[2].predict(X_test)
test_rf_model = test_rf_model[['PassengerId', 'Survived']]
#test_rf_model.to_csv('test_rf_model.csv', index=False)
print('S embarked survival %:',\
      round((train[train.Embarked == 'S']\
             .Survived.sum()/train[train.Embarked == 'S'].Survived.count())*100., 3))
print('C embarked survival %:',\
      round((train[train.Embarked == 'C']\
             .Survived.sum()/train[train.Embarked == 'C'].Survived.count())*100., 3))
print('Q embarked survival %:',\
      round((train[train.Embarked == 'Q']\
             .Survived.sum()/train[train.Embarked == 'Q'].Survived.count())*100., 3))
print(train.pivot_table(index='Pclass', columns='Embarked', values='Survived', aggfunc='count'))
print(train.pivot_table(index='Pclass', columns='Embarked', values='Survived', aggfunc='mean'))
train.pivot_table(index='Pclass', columns='Embarked', values='Survived', aggfunc='mean').plot(kind='bar')
train[train.Embarked.isnull()]
train.loc[train.Embarked.isnull(), 'Embarked'] = 'S'
train = train.join(pd.get_dummies(train.Embarked))
# doing the same for test, test does not have any null embarked values
test = test.join(pd.get_dummies(test.Embarked))
sns.distplot(train[train.Pclass == 1].Fare, label='higher class')
plt.title(train[train.Pclass == 1].Fare.describe())
plt.legend()
plt.show()
sns.distplot(train[train.Pclass == 2].Fare, label='middle class')
plt.title(train[train.Pclass == 2].Fare.describe())
plt.legend()
plt.show()
sns.distplot(train[train.Pclass == 3].Fare, label='lower class')
plt.title(train[train.Pclass == 3].Fare.describe())
plt.legend()
median_labels = [m for m in train.groupby(['Survived']).Fare.median()]
print(median_labels)
fig = plt.figure(figsize=(15,5))
ax=sns.boxplot(data=train, x='Survived', y='Fare')
median_labels = [m for m in train.groupby(['Pclass', 'Survived']).Fare.median()]
print(median_labels)
fig = plt.figure(figsize=(15,5))
ax=sns.boxplot(data=train, x='Pclass', y='Fare', hue='Survived')

plt.legend()
median_labels = [m for m in train.groupby(['Sex', 'Survived']).Fare.median()]
print(median_labels)
fig = plt.figure(figsize=(15,5))
ax=sns.boxplot(data=train, x='Sex', y='Fare', hue='Survived')

plt.legend()
test[test.Fare.isnull()]
test.loc[test.Fare.isnull(), 'Fare'] = test[(test.Pclass == 3)].Fare.median()
print(test[test.PassengerId == 1044])
train.Age.describe()
sns.distplot(train[(~train.Age.isnull()) & (train.Survived == 1)].Age, label = 'Survived')
sns.distplot(train[(~train.Age.isnull()) & (train.Survived == 0)].Age, label = 'Did Not Survive')
plt.legend()
train.loc[train.Age.isnull(), 'Age'] = -0.5
train = train.join(pd.get_dummies(pd.cut(train.Age, [-1,0,5, 12,18,35,60, 100],\
       labels=['missing', 'infant', 'child', 'teen', 'youngAdult', 'adult', 'senior'])))
train = train.join(pd.cut(train.Age, [-1,0,5, 12,18,35,60, 100],\
       labels=['missing', 'infant', 'child', 'teen', 'youngAdult', 'adult', 'senior']),\
                   rsuffix='_class')

test.loc[~test.Age.isnull(), 'Age'] = -0.5
test = test.join(pd.get_dummies(pd.cut(test.Age, [-1,0,5, 12,18,35,60, 100],\
       labels=['missing', 'infant', 'child', 'teen', 'youngAdult', 'adult', 'senior'])))
sns.barplot(data=train, x='Age_class', y='Survived')
train = train.join(pd.get_dummies(train.new_SibSp, prefix='SibSp_'))
train = train.join(pd.get_dummies(train.new_Parch, prefix='Parch_'))
test = test.join(pd.get_dummies(test.new_SibSp, prefix='SibSp_'))
test = test.join(pd.get_dummies(test.new_Parch, prefix='Parch_'))
train.columns
features = ['female', 'male',
            'Higher', 'Middle', 'Lower', 'C', 'Q', 'S',
            'missing', 'infant', 'child', 'teen', 'youngAdult', 'adult', 'senior'
            , 'SibSp__0', 'SibSp__1', 'SibSp__2', 'Parch__0', 'Parch__1', 'Parch__2']
target = ['Survived']
X_train = train[features]
y = train[target]

X_test = test[features]
from sklearn.ensemble import GradientBoostingClassifier

model_names = [
     'KNeighborsClassifier()',
    'SVC(kernel="linear")',
    'RandomForestClassifier',
    'AdaBoostClassifier',
    'GradientBoostingClassifier'
]

models = [
    KNeighborsClassifier(),
    SVC(kernel="linear"),
    RandomForestClassifier(),
    AdaBoostClassifier(),
    GradientBoostingClassifier()
]

fit_models = []

for i in range(len(models)):
    temp_score = cross_val_score(models[i], X_train, y, cv=5)
    fit_models.append(models[i].fit(X_train,y))
    print(model_names[i],' ', round(temp_score.mean(),2))
    print('+/- ',temp_score.var())
    print(metrics.classification_report(y, models[i].predict(X_train)))
print(round(metrics.accuracy_score(y, fit_models[4].predict(X_train)), 2))
sns.heatmap(metrics.confusion_matrix(y, fit_models[4].predict(X_train)), cmap='Blues', annot=True)
test_predictions_GBT = test
test_predictions_GBT['Survived'] = fit_models[4].predict(X_test)
test_predictions_GBT[['PassengerId', 'Survived']].to_csv('test_predictions_GBT.csv', index=False)
test_predictions_GBT[['PassengerId', 'Survived']].head()
