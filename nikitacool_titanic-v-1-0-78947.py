import numpy as np
import pandas as pd

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
train_data = pd.read_csv("/kaggle/input/titanic/train.csv")
test_data = pd.read_csv("/kaggle/input/titanic/test.csv")
# copy of initial datasets
train_ = train_data.copy(deep=True)
test_ = test_data.copy(deep=True)

data = pd.concat([train_, test_])
data = data.drop('Survived', axis=1)
print(data.shape)
data.head()
# Missing values?
data.isnull().sum().sum()
data['Age'].fillna(data['Age'].median(), inplace=True)
data['Fare'].fillna(data['Fare'].median(), inplace=True)
data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)
data['FamilySize'] = data['Parch']+data['SibSp']+1
    
data['IsAlone'] = 1
data['IsAlone'] = data['IsAlone'].loc[data['FamilySize']>1] = 0
    
data['Title'] = data['Name'].str.split(',', expand=True)[1].str.split(".", expand=True)[0]  
data['FareBin'] = pd.qcut(data['Fare'], 4) 
data['AgeBin'] = pd.cut(data['Age'].astype(int), 5)
stat_min = 10
title_names = (data['Title'].value_counts() < stat_min)

data['Title'] = data['Title'].apply(lambda x: 'Misc' if title_names.loc[x] == True else x)
from sklearn.preprocessing import LabelEncoder 
label = LabelEncoder()

data['Sex_Code'] = label.fit_transform(data['Sex'])
data['Embarked_Code'] = label.fit_transform(data['Embarked'])
data['Title_Code'] = label.fit_transform(data['Title'])
data['AgeBin_Code'] = label.fit_transform(data['AgeBin'])
data['FareBin_Code'] = label.fit_transform(data['FareBin'])
import matplotlib.pyplot as plt
import scipy.stats as sts
import seaborn as sns
%pylab inline
plt.style.use('fivethirtyeight')
train_ = data[:891:]
test_ = data[891::]
print(test_.shape)
print(train_.shape)
train_.head(1)
cols_1 = ['Title', 'FamilySize', 'IsAlone', 'Pclass', 'Sex']
  
plt.figure(figsize=(20,15))    
i = 0
for col in cols_1:
    i += 1
    plt.subplot(3,4,i)
    sns.countplot(train_[col], hue=train_data['Survived'])
cols_2 = ['Age', 'Fare', 'FamilySize']
  
plt.figure(figsize=(20,15))    
i = 0
for col in cols_2:
    i += 1
    plt.subplot(3,4,i)
    sns.boxplot(train_data['Survived'], train_[col])
cols_3 = ['AgeBin', 'FareBin']
  
plt.figure(figsize=(20,15))    
i = 0
for col in cols_3:
    i += 1
    plt.subplot(3,4,i)
    plt.xticks(rotation=45)
    sns.countplot(train_[col], hue=train_data['Survived'])
plt.figure(figsize=(10,7))
sns.scatterplot(train_['Age'], train_['Fare'], color='green')
plt.xlabel('Age')
plt.ylabel('Fare')
data_cols = ['Sex_Code','Pclass', 'Title_Code','SibSp', 'Parch', 'AgeBin_Code', 'FareBin_Code', 'FamilySize', 'IsAlone'] 

x = train_[data_cols]
x = pd.get_dummies(x[data_cols])
y = train_data.Survived
x.head()
from sklearn.model_selection import train_test_split 
X_train, X_valid, y_train, y_valid = train_test_split(x, y, train_size=0.8, test_size=0.2)
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import xgboost as xgb

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, cross_val_score,GridSearchCV
model = RandomForestClassifier()
model.fit(X_train, y_train)
pred = model.predict(X_valid)
accuracy_score(pred, y_valid)
model = XGBClassifier()
model.fit(X_train, y_train)
pred = model.predict(X_valid)
accuracy_score(pred, y_valid)
from sklearn import tree, linear_model, neighbors, naive_bayes, ensemble
from sklearn import model_selection

MLA = [
    #Ensemble Methods
    ensemble.AdaBoostClassifier(),
    ensemble.BaggingClassifier(),
    ensemble.GradientBoostingClassifier(),
    ensemble.RandomForestClassifier(),

    #GLM
    linear_model.SGDClassifier(),
    linear_model.Perceptron(),

    #Navies Bayes
    naive_bayes.BernoulliNB(),
    naive_bayes.GaussianNB(),

    #KNN
    neighbors.KNeighborsClassifier(),
    
    #Tree
    tree.DecisionTreeClassifier(),
    tree.ExtraTreeClassifier(),
    
    #Boosting
    XGBClassifier()
]

cv_split = model_selection.ShuffleSplit(n_splits = 5, test_size = .25, train_size = .75, random_state = 0 )
MLA_predict = X_train
row_index = 0


for alg in MLA:
    
    cv_results = model_selection.cross_validate(alg, X_train, y_train, cv = cv_split, scoring='accuracy')
    
    print(alg.__class__.__name__ + " " + str(cv_results['test_score'].mean()))
my_model = ensemble.GradientBoostingClassifier()

parameters = {'loss':('deviance', 'exponential'),
              'learning_rate':[0.001, 0.005, 0.01, 0.03, 0.05, 0.1, 0.3],
              'n_estimators':[50,100,120,150,200,300,350],
              'criterion':('friedman_mse', 'mse', 'mae'),
              'max_depth':[3,5,10,15,20,50,100]}
clf = GridSearchCV(my_model, parameters)
clf.fit(x, y)
clf.best_params_

#Output this
# {'criterion': 'mae',
#  'learning_rate': 0.005,
#  'loss': 'exponential',
#  'max_depth': 5,
#  'n_estimators': 200}
# I changed a little bit the hyperparameters
my_model_1 = ensemble.GradientBoostingClassifier(learning_rate = 0.01, loss = 'exponential',
        max_depth = 5, n_estimators = 200)
my_model_1.fit(X_train, y_train)
pred = model.predict(X_valid)
accuracy_score(pred, y_valid)
# To be continued..
test_x = test_[data_cols]
test_x = pd.get_dummies(test_x[data_cols])
test_preds = model.predict(test_x)

output = pd.DataFrame({'PassengerId': test_data.PassengerId.values,
                        'Survived': test_preds})
output.to_csv('submission.csv', index=False)