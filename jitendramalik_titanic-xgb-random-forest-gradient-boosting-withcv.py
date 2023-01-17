# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.

import pandas as pd

import csv

import numpy as np

import  re

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.model_selection import GridSearchCV,cross_val_score

from sklearn.ensemble import (GradientBoostingClassifier,RandomForestClassifier,AdaBoostClassifier,VotingClassifier)

from xgboost import XGBClassifier





#full_path = os.path.expanduser('C:\Users\jitendra\PycharmProjects\StatProject\titanic\train.csv')





train=pd.read_csv('/kaggle/input/titanic/train.csv')

test=pd.read_csv('/kaggle/input/titanic/test.csv')



PassengerId = test['PassengerId']



#print(train.head(3))

full_data=[train,test]



train["Has_Cabin"]=train["Cabin"].apply(lambda x: 0 if type(x)==float else 1)

test['Has_Cabin'] = test["Cabin"].apply(lambda x: 0 if type(x) == float else 1)



for dataset in full_data:

    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1



for dataset in full_data:

    dataset["IsAlone"] = 0

    dataset.loc[dataset['FamilySize']==1,'IsAlone']=1



#print(dataset['Embarked'])

#print(dataset['Embarked'].isnull().sum())



for dataset in full_data:

    dataset['Fare']=dataset['Fare'].fillna(value=train['Fare'].median())



train['CategoricalFare'] = pd.qcut(train['Fare'], 4)



for dataset in full_data:

    avg_age=dataset['Age'].mean()

    age_std=dataset['Age'].std()

    age_values=np.random.randint(avg_age-age_std,avg_age+age_std,size=dataset['Age'].isnull().sum())

    dataset['Age'][np.isnan(dataset['Age'])]=age_values

    dataset['Age'] = dataset['Age'].astype(int)

train['CategoricalAge'] = pd.cut(train['Age'], 5)

#print(dataset.columns)



#print(pd.crosstab(train['Title'], train['Sex']))



def get_title(name):

    title_search = re.search(' ([A-Za-z]+)\.', name)

    if title_search:

        return title_search.group()

    return ""



for dataset in full_data:

#    dataset['Title'] = dataset['Name'].apply(get_title)

    dataset['Title'] = dataset['Name'].apply(get_title)

    #print(dataset['Title'].unique())



for dataset in full_data:

    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir',

                                                 'Jonkheer', 'Dona'], 'Rare')



    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')



#print(dataset['Title'].unique())



for dataset in full_data:

    dataset['Sex']=dataset['Sex'].map({'female':0,'male':1}).astype(int)

    title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}

    dataset['Title']=dataset['Title'].map(title_mapping)

    dataset['Title']=dataset['Title'].fillna(value=0)

    dataset['Title']=dataset['Title'].astype(int)

for dataset in full_data:

    #print(dataset['Embarked'].dropna().mode()[0])

    dataset['Embarked'] = dataset['Embarked'].fillna(dataset['Embarked'].dropna().mode()[0])



for dataset in full_data:

    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)





# Mapping Fare

dataset.loc[dataset['Fare'] <= 7.91, 'Fare'] = 0

dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1

dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare'] = 2

dataset.loc[dataset['Fare'] > 31, 'Fare'] = 3

dataset['Fare'] = dataset['Fare'].astype(int)



# Mapping Age

dataset.loc[dataset['Age'] <= 16, 'Age'] = 0

dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1

dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2

dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3

dataset.loc[dataset['Age'] > 64, 'Age'] = 4



drop_elements = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'SibSp']

train = train.drop(drop_elements, axis = 1)

train = train.drop(['CategoricalAge', 'CategoricalFare'], axis = 1)

x_test  = test.drop(drop_elements, axis = 1)



x_train=train.drop(['Survived'],axis=1)

y_train = train.Survived





'''

rm=RandomForestClassifier(n_estimators=500,max_depth=6,min_samples_leaf=2)

ada=AdaBoostClassifier(n_estimators=500,learning_rate=0.75)

gb=GradientBoostingClassifier(n_estimators=500,max_depth=5,min_samples_leaf=2)

xgb=XGBClassifier(n_estimators=500,max_depth=5)





eclf = VotingClassifier(estimators=[('rm',rm), ('ada', ada), ('gb', gb),('xgb',xgb)], voting='hard')



for clf, label in zip([rm,ada, gb,xgb,eclf], ['RandomForest', 'ADAboost', 'GradientBoost', 'XtremeGradient','Combine']):

    score=cross_val_score(clf,x_train,y_train,scoring='accuracy',cv=5)

    print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (score.mean(), score.std(), label))



rm.fit(x_train,y_train)

ada.fit(x_train,y_train)

gb.fit(x_train,y_train)

xgb.fit(x_train,y_train)

eclf.fit(x_train,y_train)

prediction=eclf.predict(x_test)





submission = pd.DataFrame({

        "PassengerId": PassengerId,

        "Survived": prediction

    })





submission.to_csv("submission.csv", index=False)

'''



rm=RandomForestClassifier(max_features='auto',min_samples_leaf=2,n_estimators=50)

ada=AdaBoostClassifier(n_estimators=50,learning_rate=0.05)

gb=GradientBoostingClassifier(n_estimators=50,min_samples_leaf=2,max_features='auto')

xgb=XGBClassifier(n_estimators=50,learning_rate=0.05)





n_estimators = [int(x) for x in np.linspace(start = 50, stop = 500, num = 10)]

# Number of features to consider at every split

max_features = ['auto', 'sqrt']

# Maximum number of levels in tree

max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]

#max_depth.append(None)

# Minimum number of samples required to split a node

min_samples_split = [2, 5, 10]

# Minimum number of samples required at each leaf node

min_samples_leaf = [1, 2, 4]

# Method of selecting samples for training each tree

bootstrap = [True, False]

# Create the random grid

'''

rfrandom_grid = {'n_estimators': n_estimators,

               'max_features': max_features,

               'min_samples_split': min_samples_split,

               'min_samples_leaf': min_samples_leaf

               }

'''

learning_rate = [0.1,0.5,1.0]

#print(random_grid)

'''

adarandom_grid = {

    'n_estimators':n_estimators,

    'learning_rate': learning_rate

}



xgb_randomgrid = {

    'learning_rate' :learning_rate,

    'n_estimators':n_estimators,

    'max_depth':max_depth

}

# Random search of parameters, using 3 fold cross validation,

# search across 100 different combinations, and use all available cores

rf_random = GridSearchCV(estimator=xgb,param_grid=xgb_randomgrid,cv=5)



rf_random.fit(x_train,y_train)

print(rf_random.best_params_)



'''

eclf = VotingClassifier(estimators=[('rm',rm), ('ada', ada), ('gb', gb),('xgb',xgb)], voting='hard')



for clf, label in zip([rm,ada, gb,xgb,eclf], ['RandomForest', 'ADAboost', 'GradientBoost', 'XtremeGradient','Combine']):

    score=cross_val_score(clf,x_train,y_train,scoring='accuracy',cv=5)

    print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (score.mean(), score.std(), label))



eclf.fit(x_train,y_train)

prediction = eclf.predict(x_test)





submission = pd.DataFrame({

        "PassengerId": PassengerId,

        "Survived": prediction

    })



submission.to_csv("submission.csv", index=False)



#print(x_test.columns)