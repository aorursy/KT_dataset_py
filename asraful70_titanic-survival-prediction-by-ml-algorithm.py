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
# remove warnings
import warnings
warnings.filterwarnings('ignore')
# ---

%matplotlib inline
import pandas as pd
pd.options.display.max_columns = 100
from matplotlib import pyplot as plt
import matplotlib
matplotlib.style.use('ggplot')
import numpy as np
import seaborn as sns
pd.options.display.max_rows = 100
# Load in the train and test datasets
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
combine_data =[ train, test] # To process both set together
print(train.shape)
print(test.shape)
train.head(2) #Default is 5 row. But you can define as your wish by yourself in bracket
print(train.columns.values)
train.info()
print('*'*40)
test.info()
train.describe(include='all')
train[['Pclass', 'Survived']].groupby(['Pclass']).mean().sort_values(by='Survived', ascending=False).plot.bar()
train[["SibSp", "Survived"]].groupby(['SibSp']).mean().sort_values(by='Survived', ascending=False).plot.bar()
train[["Parch", "Survived"]].groupby(['Parch']).mean().sort_values(by='Survived', ascending=False).plot.bar()
for dataset in combine_data:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1
train[['FamilySize', 'Survived']].groupby(['FamilySize']).mean().plot.bar()
for dataset in combine_data:
    dataset.loc[dataset['FamilySize'] > 1, 'IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1
    dataset['IsAlone'] = dataset['IsAlone'].astype(int)
train[['IsAlone', 'Survived']].groupby(['IsAlone']).mean().plot.bar()
for dataset in combine_data:
    dataset['Sex'] = dataset['Sex'].map({"male": 1, "female": 0,})
    dataset['Sex'] = dataset['Sex'].astype(int)
train[["Sex", "Survived"]].groupby(['Sex']).mean().sort_values(by='Survived', ascending=False).plot.bar() 
for dataset in combine_data:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
train['Title'].value_counts()    
for dataset in combine_data:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col',\
    'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
    
train[['Title', 'Survived']].groupby(['Title']).mean().plot.bar()
title_mapping = {"Master": 1, "Miss": 2, "Mr": 3, "Mrs": 4, "Rare": 5}
for dataset in combine_data:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    
train[['Title', 'Survived']].groupby(['Title']).mean().plot.bar()    
train.Age.isnull().sum()
##########
# Easiest way to do
# train['Age']=train.Age.fillna(train.Age.mean())

###########
#To get a good assumption of missing age values
# for dataset in combine_data:
#     age_avg        = dataset['Age'].mean()
#     age_std        = dataset['Age'].std()
#     age_null_count = dataset['Age'].isnull().sum()
    
#     age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)
#     dataset['Age'][np.isnan(dataset['Age'])] = age_null_random_list
#     dataset['Age'] = dataset['Age'].astype(int)


##########
#Another way to calculate the missing Age value
for dataset in combine_data:
    dataset["Age"] = dataset.groupby(['Sex','Pclass','Title'])['Age'].transform(lambda x: x.fillna(x.median()))
sns.distplot(train['Age'])
#for dataset in combine_data:
    #dataset['Age']=pd.qcut(dataset['Age'],5,labels=[0,1,2,3,4])
#What I did below is also possible to do with above code.

for dataset in combine_data:
    dataset.loc[ dataset['Age'] <= 16, 'Age']                          = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[ dataset['Age'] > 64, 'Age']                           = 4
    dataset['Age'] = dataset['Age'].astype(int)
train[['Age', 'Survived']].groupby(['Age']).mean().plot.bar()
for dataset in combine_data:
    dataset['Fare'] = dataset['Fare'].fillna(train['Fare'].median()) #Missing value filled
    
    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare']                               = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
    dataset.loc[ dataset['Fare'] > 31, 'Fare']                                  = 3
    dataset['Fare'] = dataset['Fare'].astype(int)
    
train[['Survived','Fare']].groupby('Fare').mean().plot.bar()
#Missing value filled with maximum number of time present element.

train['Embarked'] = train['Embarked'].fillna(max(train['Embarked'].value_counts().keys())) 
#Mapping
for dataset in combine_data:
    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} )

train[['Survived','Embarked']].groupby('Embarked').mean().plot.bar()
train['Cabin'].isnull().sum()
# This column has many null values so will drop it

PassengerId_train=train['PassengerId']
PassengerId_test=test['PassengerId']

#Dropping unnecessary column
drop_columns=['Cabin','Name','Ticket','PassengerId','SibSp','Parch']
for dataset in combine_data:
    dataset.drop(drop_columns,axis=1,inplace=True)
train.head()
# Creating dummy variables 
#for dataset in combine_data:
#cat_cols = ['Pclass', 'Age', 'Fare', 'Embarked', 'Title','FamilySize']
#train= pd.get_dummies(train,columns = cat_cols,drop_first=True)
#test= pd.get_dummies(test,columns = cat_cols,drop_first=True)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score

classifiers = [
    LogisticRegression(C=0.1),
    KNeighborsClassifier(3),
    SVC(probability=True),
    BernoulliNB(),
    DecisionTreeClassifier(),
    RandomForestClassifier(n_estimators=100),
    AdaBoostClassifier(),
    GradientBoostingClassifier(),
    GaussianNB(),
    
    LinearDiscriminantAnalysis(),
    QuadraticDiscriminantAnalysis(),
    XGBClassifier()
     ]

X= train.drop("Survived", axis=1)
y = train["Survived"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1,random_state=0)
log_cols = ["Classifier", "Accuracy","F1_Accuracy"]
log      = pd.DataFrame(columns=log_cols)
for clf in classifiers:
    name = clf.__class__.__name__
    clf.fit(X_train, y_train)
    train_predictions = clf.predict(X_test)
    acc = accuracy_score(y_test, train_predictions)
    F1_acc=f1_score(y_test, train_predictions, average='weighted') #average='micro' or average='weighted'
    log_entry = pd.DataFrame([[name, acc, F1_acc]], columns=log_cols)
    log = log.append(log_entry)
log.set_index('Classifier', inplace=True)
log.sort_values(by="Accuracy",ascending=False,inplace=True)  
#log.index=log['Classifier'].values.tolist()
log.plot(kind='bar', stacked=False, figsize=(15,8))
log
#Feature Selection : RandomForestClassifier example is used
#X_train= train.drop("Survived", axis=1)
#y_train = train["Survived"]
#X_test=test
#from sklearn.feature_selection import SelectFromModel
#parameters = {'bootstrap': False, 'max_depth': 6, 'max_features': 'log2', 'min_samples_leaf': 1,
#                                             'min_samples_split': 3, 'n_estimators': 50}
#clf=RandomForestClassifier(**parameters )
#clf.fit(X_train, y_train)
#survived = clf.predict(X_test)
#features = pd.DataFrame()
#features['feature'] = X_train.columns
#features['importance'] = clf.feature_importances_
#features.sort_values(by=['importance'], ascending=True, inplace=True)
#features.set_index('feature', inplace=True)
#features.plot(kind='barh', figsize=(20, 20))

#model = SelectFromModel(clf, prefit=True)
#train_reduced = model.transform(X_train)
#test_reduced=model.transform(X_test)

#clf=RandomForestClassifier(**parameters )
#clf.fit(train_reduced, y_train)
#survived = clf.predict(test_reduced)
#submission=pd.DataFrame({'PassengerId':PassengerId_test,'survived':survived})
#submission.to_csv('gender_submission4.csv',index=False )
#parameter_grid = {
#                 'max_depth' : [4, 6, 8],
#                 'n_estimators': [50, 10],
#                 'max_features': ['sqrt', 'auto', 'log2'],
#                 'min_samples_split': [2, 3, 8],
 #                'min_samples_leaf': [1, 3, 10],
#                 'bootstrap': [True, False],
#                 }
#forest = RandomForestClassifier()
#grid_search = GridSearchCV(forest,
#                               scoring='accuracy',
#                               param_grid=parameter_grid,
#                               cv=5)

#grid_search.fit(X_train, y_train)
#model = grid_search
#parameters = grid_search.best_params_

#print('Best score: {}'.format(grid_search.best_score_))
#print('Best parameters: {}'.format(grid_search.best_params_))

#acc = accuracy_score(y_test, survived)
#F1_acc=f1_score(y_test, survived, average='macro') 
