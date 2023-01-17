# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import re

def get_title(name):
    title_search = re.search(' ([A-Za-z]+)\.', name)
    
    if title_search:
        return title_search.group(1)
    else:
        return 'None'

def preprocess(df):
    df['Title'] = df['Name'].apply(get_title)
    df['Title'] = df['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    df['Title'] = df['Title'].replace('Mlle', 'Miss')
    df['Title'] = df['Title'].replace('Ms', 'Miss')
    df['Title'] = df['Title'].replace('Mme', 'Mrs')
    
    df['Age'].fillna(-0.5, inplace = True)
    # 年齢の階層化
    df['Age_bin'] = pd.cut(df['Age'], bins=[-1,0,5,12,20,40,120], labels=['Unkown','Baby','Children','Teenage','Adult','Elder'])
    df.drop(['Age'], axis=1, inplace=True)
    
    #df['Cabin'].fillna('N', inplace=True)
    #df['Cabin'] = df['Cabin'].apply(lambda x: x[0])
    
    # Embarkedの欠損値補填
    df['Embarked'].fillna('C', inplace = True)
    # Fareの欠損値補填
    #df['Fare'].fillna(-0.5, inplace = True)
    # 料金の階層化
    #df['Fare_bin'] = pd.cut(df['Fare'], bins=[-1,0,7.91,14.45,31,120], labels=['Unkown','Low_fare','median_fare','Average_fare','high_fare'])
    df.drop(['Fare'], axis=1, inplace=True)
    
    # 家族サイズ
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['FamilySize_bin'] = pd.cut(df['FamilySize'], bins=[0,1,2,4,20], labels=['Single','Small','Large','Fuge'])
    df.drop(['FamilySize'], axis=1, inplace=True)
    
    df['SibSp_bin'] = pd.cut(df['SibSp'], bins=[0,1,5,20], labels=['None','Small','Large'], right=False)
    df.drop(['SibSp'], axis=1, inplace=True)
    df['Parch_bin'] = pd.cut(df['Parch'], bins=[0,1,4,20], labels=['None','Small','Large'], right=True)
    df.drop(['Parch'], axis=1, inplace=True)
    
    # 分類(or階層)パラメータの数値化
    df = pd.get_dummies(
        df,
        columns=['Pclass','Sex','Embarked','Age_bin','FamilySize_bin','SibSp_bin','Parch_bin','Title'],
        prefix=['Pclass','Sex','Embarked','Age_bin','FamilySize_bin','SibSp_bin', 'Parch_bin','Title'],
        drop_first=False)

    #扱いにくいカラムの削除
    df.drop(['Name','Ticket','Cabin'],axis=1,inplace=True)
    
    df.drop(['Pclass_2','Sex_male','Embarked_Q','Embarked_C','Age_bin_Children','Age_bin_Teenage','SibSp_bin_None','SibSp_bin_Small','SibSp_bin_Large','Parch_bin_Large','Title_Mr','Title_Rare','FamilySize_bin_Single'], axis=1,inplace=True)
    
    return df
def load_data():
    train_df = pd.read_csv('../input/train.csv')
    test_df = pd.read_csv('../input/test.csv')
    
    return preprocess(train_df), preprocess(test_df)
train_df, test_df = load_data()
train_df.drop(['PassengerId'],axis=1).corr()
#print(train_df.values.shape[1])
#test_df.head()
sns.heatmap(train_df.drop(['PassengerId'],axis=1).corr(),annot=True,cmap='RdYlGn',linewidths=0.2)
fig=plt.gcf()
fig.set_size_inches(20,12)
plt.show()
def features_and_label(train_df):
    label = train_df["Survived"].values
    features = train_df.drop(['PassengerId','Survived'],axis=1).values
    
    return features,label

X_train, y_train = features_and_label(train_df)

print(X_train)
print(y_train)
from sklearn.model_selection import GridSearchCV

from sklearn.svm import SVC

param_grid = {'kernel': ['rbf'], 
                  'gamma': [ 0.001, 0.01, 0.1, 1],
                  'C': [1, 10, 50, 100,200,300, 1000]}

modelsvm = GridSearchCV(SVC(),param_grid = param_grid, cv=5, scoring="accuracy", n_jobs= 4, verbose = 1)

modelsvm.fit(X_train,y_train)

print(modelsvm.best_estimator_)

# Best score
print(modelsvm.best_score_)
from sklearn.ensemble import RandomForestClassifier

n_estim=range(100,1000,100)

## Search grid for optimal parameters
param_grid = {"n_estimators" :n_estim}


model_rf = GridSearchCV(RandomForestClassifier(),param_grid = param_grid, cv=5, scoring="accuracy", n_jobs= 4, verbose = 1)

model_rf.fit(X_train,y_train)



# Best score
print(model_rf.best_score_)

#best estimator
model_rf.best_estimator_
from sklearn.neural_network import MLPClassifier

param_grid = {
    "hidden_layer_sizes":[(100,),(100, 10),(100, 100, 10),(100,100,50,10)],
    "max_iter":[1000],
    "early_stopping":[True],
    "batch_size":[20,50,100,200],
    "alpha":[0.0001,0.0005,0.001]
}

model_nn = GridSearchCV(MLPClassifier(),param_grid = param_grid, cv=5, scoring="accuracy", n_jobs= 4, verbose = 1)

model_nn.fit(X_train,y_train)



# Best score
print(model_nn.best_score_)

#best estimator
model_nn.best_estimator_
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import cross_validate

X_test = test_df.drop(['PassengerId'],axis=1).values
X_test = X_test

estimators = list(zip(['svm','rf','nn'],[modelsvm.best_estimator_,model_rf.best_estimator_,model_nn.best_estimator_]))
clf = VotingClassifier(estimators,voting='hard')

clf.fit(X_train,y_train)
print(clf.score(X_train,y_train))
print(cross_validate(clf,X_train,y_train,cv=5))

y_pred = clf.predict(X_test)

submission = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Survived": y_pred})

print(submission)

submission.to_csv("titanic_submission.csv", index=False)
