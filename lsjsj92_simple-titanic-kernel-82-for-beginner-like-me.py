import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('fivethirtyeight')
import warnings
warnings.filterwarnings('ignore')
%matplotlib inline
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
datas = [train, test]
for data in datas:
    print(data.isnull().sum())
train['Survived'].value_counts().plot.bar()
sns.countplot('Sex', hue='Survived', data=train)
sns.countplot('Pclass', hue='Survived', data=train)
sns.factorplot('Pclass', 'Survived', hue='Sex', data=train)
f, ax = plt.subplots(1,2,figsize=(12,6))
sns.violinplot('Pclass', 'Age', hue='Survived', data=train, split=True, ax=ax[0])
ax[0].set_yticks(range(0,90,10))

sns.violinplot('Sex', 'Age', hue='Survived', data=train, split=True, ax=ax[1])
ax[1].set_yticks(range(0, 90, 10))
sns.factorplot('Embarked', 'Survived', data=train)
sns.factorplot('Embarked', 'Survived', hue='Sex', col='Pclass', data=train)
sns.barplot('SibSp', 'Survived', data=train)
sns.barplot('Parch', 'Survived', data=train)
del train, test, datas
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
passenger = test['PassengerId']
datas = [train, test]
for data in datas:
    data['Initial'] = 0
    data['Initial'] = data['Name'].str.extract('([A-Za-z]+)\.')
train['Initial'].unique()
for data in datas:
    data['Initial'].replace(['Mlle','Mme','Ms','Dr','Major','Lady','Countess','Jonkheer','Col','Rev','Capt','Sir','Don', 'Dona'],['Miss','Miss','Miss','Mr','Mr','Mrs','Mrs','Other','Other','Other','Mr','Mr','Mr','Other'],inplace=True)
train.groupby('Initial')['Age'].mean()
for data in datas:
    data.loc[(data['Age'].isnull()) & (data['Initial'] == 'Mr'), 'Age'] = 33
    data.loc[(data['Age'].isnull()) & (data['Initial'] == 'Master'), 'Age'] = 5
    data.loc[(data['Age'].isnull()) & (data['Initial'] == 'Mrs'), 'Age'] = 36
    data.loc[(data['Age'].isnull()) & (data['Initial'] == 'Miss'), 'Age'] = 22
    data.loc[(data['Age'].isnull()) & (data['Initial'] == 'Other'), 'Age'] = 46
print(datas[0]['Age'].isnull().sum())
print(datas[1]['Age'].isnull().sum())
print(datas[0]['Initial'].unique())
print(datas[1]['Initial'].unique())
for data in datas:
    data['Embarked'].fillna('S', inplace=True)
print(datas[0]['Embarked'].isnull().sum())
print(datas[1]['Embarked'].isnull().sum())
for data in datas:
    data['Fare'] = data["Fare"].map(lambda i: np.log(i) if i > 0 else 0)
for data in datas:
    data['Age_band']=0
    data.loc[data['Age']<=16,'Age_band']=0
    data.loc[(data['Age']>16)&(data['Age']<=32),'Age_band']=1
    data.loc[(data['Age']>32)&(data['Age']<=48),'Age_band']=2
    data.loc[(data['Age']>48)&(data['Age']<=64),'Age_band']=3
    data.loc[data['Age']>64,'Age_band']=4
for data in datas:
    data['Family_Size']=0
    data['Family_Size']=data['Parch']+data['SibSp']
    data['Alone']=0
    data.loc[data.Family_Size==0,'Alone']=1
for data in datas:
    data['Sex'].replace(['male','female'],[0,1],inplace=True)
    data['Embarked'].replace(['S','C','Q'],[0,1,2],inplace=True)
    data['Initial'].replace(['Mr','Mrs','Miss','Master','Other'],[0,1,2,3,4],inplace=True)
for data in datas:
    data.drop(['Name','Age','Ticket','Cabin','PassengerId'],axis=1,inplace=True)
train.columns
test.columns
train.head()
X=train[train.columns[1:]]
Y=train['Survived']
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve

kfold = StratifiedKFold(n_splits=10)
decisionTree = DecisionTreeClassifier()

ada = AdaBoostClassifier(decisionTree, random_state=0)

## 그리드 서치
ada_param_grid = {"base_estimator__criterion" : ["gini", "entropy"],
              "base_estimator__splitter" :   ["best", "random"],
              "algorithm" : ["SAMME","SAMME.R"],
              "n_estimators" :[1,2,3],
              "learning_rate":  [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3,1.5]}

decision_grid = GridSearchCV(ada, param_grid = ada_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)

decision_grid.fit(X,Y)

ada_best = decision_grid.best_estimator_
decision_grid.best_score_
rf = RandomForestClassifier()


## 그리드 서치
rf_param_grid = {"max_depth": [None],
              "max_features": [1, 3, 10],
              "min_samples_split": [2, 3, 10],
              "min_samples_leaf": [1, 3, 10],
              "bootstrap": [False],
              "n_estimators" :[100,300, 500, 700, 800, 900]
              }


rf_grid = GridSearchCV(rf,param_grid = rf_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)

rf_grid.fit(X,Y)

rf_best = rf_grid.best_estimator_
rf_grid.best_score_
gradient_boost = GradientBoostingClassifier()
gb_param_grid = {'loss' : ["deviance"],
              'n_estimators' : [100,200,300],
              'learning_rate': [0.1, 0.05, 0.01],
              'max_depth': [4, 8],
              'min_samples_leaf': [100,150,200],
              'max_features': [0.3, 0.1] 
              }

gb_grid = GridSearchCV(gradient_boost,param_grid = gb_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)

gb_grid.fit(X,Y)

gb_best = gb_grid.best_estimator_
gb_grid.best_score_
svc = SVC(probability=True)
svc_param_grid = {'kernel': ['rbf'], 
                  'gamma': [ 0.001, 0.01, 0.1, 0.5, 1],
                  'C': [0.01, 0.1, 1, 10, 50, 100,200,300]}

svc_grid = GridSearchCV(svc,param_grid = svc_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)

svc_grid.fit(X,Y)

svc_best = svc_grid.best_estimator_
svc_grid.best_score_
vote = VotingClassifier(estimators=[('rf', rf_best), ('ada', ada_best),
('svc', svc_best), ('gb',gb_best)], voting='soft', n_jobs=4)

vote_result = vote.fit(X,Y)
pred = vote.predict(test)
#submission = pd.DataFrame({
#    'PassengerId' : passenger,
#    'Survived' : pred
#})
#submission.head()
#import os

#if os.path.exists("./submission.csv"):os.remove("./submission.csv")
#print(os.listdir("./"))
#submission.to_csv('submission.csv', index=False)
#print(os.listdir("./"))
