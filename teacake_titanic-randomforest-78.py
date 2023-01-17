import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

import seaborn as sns

from sklearn import tree

from sklearn import preprocessing

from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split

from sklearn.linear_model import SGDClassifier

from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import VotingClassifier

from sklearn import svm, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, gaussian_process

from sklearn.ensemble import RandomForestClassifier

from sklearn import feature_selection

from sklearn import model_selection

from sklearn import metrics

from xgboost import XGBClassifier

from sklearn.model_selection import cross_val_score

from sklearn.metrics import make_scorer, accuracy_score

# Figures inline and set visualization style

%matplotlib inline

sns.set()

import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data_test = pd.read_csv("../input/test.csv")

train = pd.read_csv("../input/train.csv")
sns.countplot(x='Survived', data=train);

sns.countplot(x='Sex', data=train);

sns.catplot(x='Survived', col='Sex', kind='count', data=train);

#Drop coloums that are not needed

train.drop('Name',axis = 1, inplace =True)

train.drop('Ticket', axis = 1, inplace = True)

data_test.drop('Name',axis = 1, inplace =True)

data_test.drop('Ticket', axis = 1, inplace = True)
# feature scale sex column 

m = {'m' : 1, 'f' : 0}

train['Sex'] = train['Sex'].str[0].str.lower().map(m)

data_test['Sex'] = data_test['Sex'].str[0].str.lower().map(m)
# feature scale Embarked

em = {'S':0, 'C': 1, 'Q' :2}

train['Embarked'] = train['Embarked'].str[0].str.upper().map(em)

train['Embarked'] = train['Embarked'].fillna(1)

data_test['Embarked'] = data_test['Embarked'].str[0].str.upper().map(em)

data_test['Embarked'] = data_test['Embarked'].fillna(1)
# filled missing ages with the average age 

train['Age'] = train['Age'].fillna(train['Age'].median())

data_test['Age'] = train['Age'].fillna(train['Age'].median())
# Replace missing values of fare with the average 

train['Fare'] = train['Fare'].fillna(train['Fare'].median())

data_test['Fare'] = train['Fare'].fillna(train['Fare'].median())
# change age and fare types to int

train['Age'] = train['Age'].astype('int64')

data_test['Age'] = data_test['Age'].astype('int64')

train['Fare'] = train['Fare'].astype('int64')

data_test['Fare'] = data_test['Fare'].astype('int64')
# drop cabin

train.drop('Cabin', axis = 1, inplace = True)

data_test.drop('Cabin', axis = 1, inplace = True)
# corrilation heat map of data after feature scaleing 

corr = train.corr()

sns.heatmap(corr, 

        xticklabels=train.columns,

        yticklabels=train.columns)
'''

# normlising coloumns 

train=(train-train.min())/(train.max()-train.min())

temp = data_test['PassengerId']

data_test = (data_test-data_test.min())/(data_test.max()-data_test.min())

data_test['PassengerId'] = temp

'''
# seperating the data for training 

X_all = train.drop(['Survived', 'PassengerId'], axis=1)

y_all = train['Survived']



num_test = 0.20

X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=num_test, random_state=23, shuffle = True)
'''

# testing models 

model  = [

    ensemble.AdaBoostClassifier(),

    ensemble.BaggingClassifier(),

    ensemble.ExtraTreesClassifier(),

    ensemble.GradientBoostingClassifier(),

    ensemble.RandomForestClassifier(),

    gaussian_process.GaussianProcessClassifier(),

    linear_model.LogisticRegressionCV(),

    naive_bayes.BernoulliNB(),

    naive_bayes.GaussianNB(),

    tree.DecisionTreeClassifier(),

    XGBClassifier(),

    SGDClassifier()

    ]

for i in model:

    print(str(i) + str(cross_val_score(i,X_test, y_test, scoring = "accuracy", cv = 10)))

'''

'''

# gridsearch for paramaters 



rfc = XGBClassifier() 



param_grid = {"learning_rate"    : [0.01, 0.05] ,

              "max_depth"        : [ 3],

              "gamma"            : [ 0.1 ],

              "colsample_bytree" : [ 0.5, 0.7 ],

              'n_estimators': [1000],

              'subsample': [0.4]

              

}



CV_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv= 5, scoring="accuracy")

CV_rfc.fit(X_train, y_train)

print(CV_rfc.best_params_)



'''
'''

# fine tune paramaters 

cross_val = cross_val_score(ensemble.RandomForestClassifier(random_state = 90, warm_start = True, 

                                  min_samples_leaf = 1,

                                  min_samples_split = 2,

                                  n_estimators = 20,

                                  max_depth = 5, 

                                  max_features = 'sqrt'), X_test, y_test, cv=100)

print(np.mean(cross_val))

print(np.std(cross_val))

'''


model = (ensemble.RandomForestClassifier(warm_start = True, 

                                  min_samples_leaf = 1,

                                  min_samples_split = 2,

                                  n_estimators = 26,

                                  max_depth = 6, 

                                  max_features = 'sqrt'))

model.fit(X_all,y_all)

model.score(X_test,y_test)

'''

model1 =  ensemble.RandomForestClassifier(random_state = 10, warm_start = True, 

                                  n_estimators = 26,

                                  max_depth = 6, 

                                  max_features = 'sqrt')

model2 =  ensemble.GradientBoostingClassifier(criterion = 'friedman_mse',

                                              learning_rate = 0.075,

                                              loss = 'deviance',

                                              max_depth = 5,

                                              max_features = 'sqrt',

                                              n_estimators = 10,

                                              subsample = 0.9)

model3 = XGBClassifier(colsample_bytree = 0.7,

                       gamma = 0.1, 

                       learning_rate = 0.05, 

                       max_depth = 3,

                       n_estimators = 1000,

                       subsample = 0.4)



model = VotingClassifier(estimators=[('rf', model1), ('Gb', model2), ('XGB', model3)], voting='soft')

model.fit(X_all,y_all)

model.score(X_test,y_test)

'''
# useing model to make predictions and making the submission file 

test_pre = model.predict(data_test.drop(['PassengerId'], axis=1))

data_test["Survived"] = test_pre.astype(int)

data_test[['PassengerId', 'Survived']].to_csv('submission.csv', index = False)