import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
% matplotlib inline
import re
from pandas import set_option

import warnings
warnings.filterwarnings('ignore')

plt.rcParams['patch.force_edgecolor'] = True
plt.rcParams['patch.facecolor'] = 'b'
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")

train.head(3)
print(train.info())
set_option('precision',2) # set numbers to 2 decimal place
print(train.describe())
print(train['Survived'].value_counts()/train['PassengerId'].count()*100)
plt.figure(figsize=(9,5))
set_option('precision', 3)

sns.heatmap(train.corr(), cmap = 'YlOrRd', annot=True, linewidths=2)
plt.title('Correlation between features')
plt.tight_layout()
features = ['Age','Fare','Parch','SibSp']

plt.plot(train[features].skew(), 'ro', alpha=0.6)
plt.title('Numerical features skewness')
plt.xlabel('Features')
plt.ylabel('Skewness (higher indicates more skewed)')
plt.show()
# keeping similar ratio of survived in train and validation set
from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=42)

for train_index, validation_index in split.split(train, train['Survived']):
    train_set = train.loc[train_index]
    valid_set = train.loc[validation_index]
    
X_train = train_set.drop('Survived', axis=1)
y_train = train_set['Survived']

X_valid = valid_set.drop('Survived', axis=1)
y_valid = valid_set['Survived']

X_test = test # 'test' set is imported earlier
fig, ax = plt.subplots(2,3, figsize=(18,9))

# pie chart of overall survival rate
survival_rate = train_set['Survived'].value_counts()/len(train_set['Survived'])
labels=['Not survived', 'Survived']
colors = ['lightcoral','gold']
explode = (0.1,0)

ax[0,0].pie(survival_rate, labels=labels, colors=colors, explode=explode, autopct='%1.1f%%', shadow=True)
ax[0,0].axis('equal')
ax[0,0].set_title('Overall survival rate')

# bar chart of survival by gender
survived = train_set[train_set['Survived']==1][['Sex','Survived']].groupby('Sex')['Survived'].count().tolist()
not_survived = train_set[train_set['Survived']==0][['Sex','Survived']].groupby('Sex')['Survived'].count().tolist()

N = 2
width = 0.6
x = np.arange(N)

ax[0,1].bar(x, survived, width=width, color='gold', alpha=0.6, label='Survived')
ax[0,1].bar(x, not_survived, width=width, color='red', bottom=survived, alpha=0.5, label='Not survived')
ax[0,1].set_xticks(x)
ax[0,1].set_xticklabels(['Female','Male'])
ax[0,1].set_ylabel('Number of passengers')
ax[0,1].set_title('Survival by Gender')
ax[0,1].legend(loc='upper left')

# Survival by class
survival_class = train_set[['Pclass','Survived']].groupby(['Pclass'],as_index=False).mean()
ax[0,2].bar(survival_class['Pclass'], survival_class['Survived'], color='gold', alpha=0.6)
y = survival_class['Survived'].tolist()
for i, v in enumerate(y) :
    ax[0,2].text(i+0.9, v+0.02, str("{0:.2f}".format(v)))
ax[0,2].set_yticks([0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7])
ax[0,2].set_xticks(train_set['Pclass'])
ax[0,2].set_ylabel('Survival rate')
ax[0,2].set_xlabel('Pclass')
ax[0,2].set_title('Survival by class')

# Survival by Age
sns.distplot(train_set[train_set['Survived']==0]['Age'].dropna(),ax=ax[1,0],kde=False,color='red',bins=20)
sns.distplot(train_set[train_set['Survived']==1]['Age'].dropna(),ax=ax[1,0],kde=False,color='yellow',bins=20)
ax[1,0].legend(['Not survived','Survived'])
ax[1,0].set_title('Survival by age')
ax[1,0].set_ylabel('Number of passengers')

# Survival by port of embarkation
sns.countplot('Embarked', hue='Survived',data=train_set,ax=ax[1,1], palette=['red','gold'], alpha=0.5)
ax[1,1].legend(['Not survived','Survived'])
ax[1,1].set_title('Survival by embarked port')

# Survival by Parch
sns.countplot('Parch', hue='Survived', data=train_set, ax=ax[1,2], palette=['red','gold'], alpha=0.5)
ax[1,2].legend(['Not Survived','Survived'], loc='upper right')
ax[1,2].set_xlabel('sibsp (# of siblings / spouses aboard)')

plt.tight_layout()
plt.show()
train_set.isnull().sum() # quantify missing data in features

# Missing data of Age will be filled with median, Cabin features to be dropped and Embarked to be filled with most frequent
full_data = [X_train, X_valid, X_test]
# Filling missing Age data
for dataset in full_data:
    dataset['Age'] = dataset['Age'].fillna(dataset['Age'].median())

# Filling Embarked with most frequent port
for dataset in full_data:
    dataset['Embarked'] = dataset['Embarked'].fillna(dataset['Embarked'].value_counts().index[0])
# Adding family size feature
for dataset in full_data:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1

# Define function to extract titles from passenger names
def get_title(name):
    title_search = re.search(' ([A-Za-z]+)\.', name)
    # If the title exists, extract and return it
    if title_search:
        return title_search.group(1)
    return ""
# Create a new feature Title, containing the titles of passenger names
for dataset in full_data:
    dataset['Title'] = dataset['Name'].apply(get_title)
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 
                                           'Sir', 'Jonkheer', 'Dona'], 'Rare')
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
for dataset in full_data:
    dataset['Sex'] = dataset['Sex'].map({'male':0,'female':1})
    dataset['Embarked'] = dataset['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
    dataset['Title'] = dataset['Title'].map({"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5})

    # range can be retrieved from pd.qcut(train['Fare'], 4).value_counts()
    dataset['Fare'] = dataset['Fare'].fillna(dataset['Fare'].median())
    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
    dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

for dataset in full_data:
    dataset[['Age','SibSp','Parch','FamilySize']] = scaler.fit_transform((dataset[['Age','SibSp','Parch','FamilySize']]))
selected_features = ['Pclass','Sex','Age','Fare','Embarked','Title','FamilySize','SibSp','Parch']

X_train_selected = X_train[selected_features]
X_valid_selected = X_valid[selected_features]
X_test_selected = X_test[selected_features]
X_train_selected.head()
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, KFold
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

folds = 10
seed = 42
scoring = 'accuracy'

models = []
models.append(('LR', LogisticRegression()))
models.append(('SVC', SVC()))
models.append(('RF', RandomForestClassifier()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('Ada', AdaBoostClassifier()))
models.append(('DT', DecisionTreeClassifier()))
models.append(('XGB', XGBClassifier()))
models.append(('LGBM', LGBMClassifier()))

results = []
names = []

for name, model in models:
    kfold = KFold(n_splits=folds, random_state=seed)
    cv_results = cross_val_score(model, X_train_selected, y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)
fig, ax = plt.subplots(1,1, figsize=(10,6))

ax.boxplot(results)
ax.set_title('Models performance comparison')
ax.set_xticklabels(names)

print('KNN, LGBM and XGB seems to be performing better')
plt.show()
from sklearn.model_selection import GridSearchCV

param_grid_knn = [{'n_neighbors':[2,4,8,16,32],'p':[2,3]}]

grid_knn = GridSearchCV(KNeighborsClassifier(), param_grid_knn, cv=10, scoring='accuracy')
grid_knn.fit(X_train_selected, y_train)
optimized_knn = grid_knn.best_estimator_
print(grid_knn.best_params_)
param_grid_xgb = [{'learning_rate':[0.1,0.5,1.0],
                   'reg_alpha':[0,0.5,1.0],
                   'reg_lambda':[0,0.5,1.0],
                   'gamma':[0.1,0.5,0.9],
                   'min_child_weight':[3,5,7],
                   'subsample':[0.7,1.0],
                   'colsample_bytree':[0.5,1.0]}]

grid_xgb = GridSearchCV(XGBClassifier(), param_grid_xgb, cv=10, scoring='accuracy')
grid_xgb.fit(X_train_selected, y_train)
optimized_xgb = grid_xgb.best_estimator_
print(grid_xgb.best_params_)
param_grid_lgbm = {
    'learning_rate': [0.1,0.5,1.0],
    'colsample_bytree':[0.5,1.0],
    'subsample':[0.6,0.7,0.9,1.0],
    'reg_alpha':[0.0,0.5,1.0],
    'reg_lambda':[0.0,0.5,1.0]}

grid_lgb = GridSearchCV(LGBMClassifier(), param_grid_lgbm, cv=10, scoring='accuracy')
grid_lgb.fit(X_train_selected, y_train)
optimized_lgb = grid_lgb.best_estimator_
print(grid_lgb.best_params_)
models_op = []

models_op.append(('knn_optimized', optimized_knn))
models_op.append(('xgb_optimized', optimized_xgb))
models_op.append(('lgbm_optimized', optimized_lgb))

results_op = []
names_op = []

for name, model in models_op:
    predictions = model.predict(X_valid_selected)
    accuracy = accuracy_score(y_valid, predictions)
    results_op.append(accuracy)
    names_op.append(name)
    msg = "%s: %f" % (name, accuracy)
    print(msg)
test_id = test['PassengerId']

test_predict_knn = optimized_knn.predict(X_test_selected)
prediction_submission_knn = pd.DataFrame({'PassengerId':test_id,'Survived':test_predict_knn})
prediction_submission_knn.to_csv('titanic_knn.csv', index=False)