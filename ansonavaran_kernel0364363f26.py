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
data_train = pd.read_csv('../input/train.csv')

data_test  = pd.read_csv('../input/test.csv')
type(data_test)
import os

from sklearn import preprocessing

from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV

from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import ExtraTreesClassifier

from sklearn.linear_model import RidgeClassifierCV

from sklearn.ensemble import AdaBoostClassifier

from sklearn.ensemble import BaggingClassifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import accuracy_score,f1_score

import numpy as np
data1 = data_train.copy(deep =True)

data2 = data_test.copy(deep =True)





frame = [data1,data2]



lis=[]

lis1=[]

print(data1.info())

print(data2.info())





for col in data1:

    if(data1[col].isnull().any()):

        lis.append(col)

        

for col in data2:

    if(data2[col].isnull().any()):

        lis1.append(col) 

        

lis.remove('Cabin') 

lis1.remove('Cabin')   

      



data1['Age'].fillna(data1['Age'].median(),inplace =True)

data1['Embarked'].fillna(data1['Embarked'].mode()[0],inplace =True)



data2['Age'].fillna(data2['Age'].median(),inplace =True)

data1['Fare'].fillna(data1['Fare'].median(),inplace =True)

data2['Fare'].fillna(data2['Fare'].median(),inplace =True)





#drpingg values



drop_val = ['PassengerId','Cabin', 'Ticket']

data1.drop(drop_val,axis=1,inplace = True)



#feature adding new

for dataset in frame:    

    

    dataset['FamilySize'] = dataset ['SibSp'] + dataset['Parch'] + 1



    dataset['IsAlone'] = 1 

    dataset['IsAlone'].loc[dataset['FamilySize'] > 1] = 0

    

    

label = preprocessing.LabelEncoder()  





for d in frame:

    

    d['Sex'] = label.fit_transform(d['Sex'])

    d['Embarked'] = label.fit_transform(d['Embarked'])

    

    

    

tar = ['Survived']   

x_label = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare',

        'Embarked', 'FamilySize', 'IsAlone'] 





data1.drop(['Name'],axis=1,inplace=True)



train_x, test_x, train_y, test_y = train_test_split(data1[x_label], data1[tar],test_size=0.1, random_state = 0)

data1.isnull().sum()
import xgboost 

from sklearn.ensemble import ExtraTreesClassifier

from xgboost import XGBClassifier
classes = [SVC(),RandomForestClassifier(),AdaBoostClassifier(),BaggingClassifier(),GradientBoostingClassifier(),GaussianNB(),XGBClassifier(),ExtraTreesClassifier()]

#classes1 = [AdaBoostClassifier(),BaggingClassifier(),GradientBoostingClassifier(),GaussianNB(),XGBClassifier(),ExtraTreesClassifier()]





params = [[{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],

                     'C': [1, 10, 100, 1000]},

                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}],[{ 

    'n_estimators': [100,200, 400],

    'max_features': ['auto', 'sqrt', 'log2'],

    'max_depth' : [4,5,6,7,8],

    'criterion' :['gini', 'entropy']}],

 [ {'n_estimators': [100,200,300,400,500],

'learning_rate':[0.01,0.1,1]}],[{'n_estimators': [100,200,300,400,500],

'max_samples':[.5, .75, 1.0]}] ,[{"loss":["deviance"],"n_estimators":[50,100,300,400,500],

"max_depth":[3,5,8]}] ,[{"priors":[None]}],[{'learning_rate':[.03, .05],'max_depth': [1,2,4,6,8,10],'n_estimators':[ 50, 100, 300,500]}    ],

 [{'n_estimators':[100,200,400,500],'criterion':['gini', 'entropy'],'max_depth':[1,2,4,6,8,10]}]]



"""params1=[[ {'n_estimators': [100,200,400],

'learning_rate':[0.01,0.1,1]}],[{'n_estimators': [100,200,400],

'max_samples':[.5, .75, 1.0]}] ,[{"loss":["deviance"],"n_estimators":[50,100,300],

"max_depth":[3,5,8]}] ,[{"priors":[None]}],[{'learning_rate':[.03, .05],'max_depth': [1,2,4,6,8,10],'n_estimators':[10, 50, 100, 300]}    ],

 [{'n_estimators':[100,200,400],'criterion':['gini', 'entropy'],'max_depth':[1,2,4,6,8,10]}]]

"""





testlist=[]

paramslist=[]

bestscore=[]

bestestimator =[]



for c,p in zip(classes,params):

    

    print("the model using  is {}".format(c))

    print('\n')

    print('\n')

    grid_search = GridSearchCV(estimator=c,param_grid=p,cv= 5,n_jobs=-1)

    grid_search.fit(train_x,train_y.values.ravel())

    best_param = grid_search.best_params_

    print(best_param)

    paramslist.append(best_param)

    

    best_score=grid_search.best_score_

    bestscore.append(best_score)

    

    best_estimator=grid_search.best_estimator_

    bestestimator.append(best_estimator)

    ypred= best_estimator.predict(test_x)

    testlist.append(tuple((accuracy_score(test_y,ypred),f1_score(test_y,ypred))))
data1

from sklearn.ensemble import VotingClassifier



voting = VotingClassifier(estimators = [('svm', bestestimator[0]),('rf', best_estimator[1]),('ad', bestestimator[2]),('bag', bestestimator[3]),('grad', best_estimator[4]),('gauss', bestestimator[5]),('xgb', bestestimator[6]),('extra', bestestimator[7])],voting = 'hard') 

#classes = [SVC(),RandomForestClassifier(),AdaBoostClassifier(),BaggingClassifier(),GradientBoostingClassifier(),GaussianNB(),XGBClassifier(),ExtraTreesClassifier()]
data3=data2.copy(deep=True)

data3.drop(['Name','Ticket','Cabin'],axis=1,inplace=True)

data3.drop(['PassengerId'],axis=1,inplace=True)

yy=bestestimator[6].predict(data3)

submission = pd.DataFrame({'PassengerId':data2['PassengerId'],'Survived':yy})

submission.head()





filename = 'Titanic Predictions 2.csv'



submission.to_csv(filename,index=False)



print('Saved file: ' + filename)