# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.



import seaborn as sns

%matplotlib inline
df_train = pd.read_csv('../input/train.csv')

df_test = pd.read_csv('../input/test.csv')
df_train['train'] = 1

df_test['train'] = 0



df = pd.concat([df_test, df_train])





survived = df_train['Survived'].copy()

df = df.drop('Survived', axis = 1)
df_train.head()
df_train.describe()
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1



df['Name_length'] = df['Name'].apply(len)



df['IsAlone'] = 0



df.loc[df['FamilySize'] == 1, 'IsAlone'] = 1



df['Title'] = 0



df['Title'] = df.Name.str.extract('([A-Za-z]+)\.') 



df['Title'].replace(['Mlle','Mme','Ms','Dr','Major','Lady','Countess','Jonkheer','Col',

                         'Rev','Capt','Sir','Don'], ['Miss','Miss','Miss','Mr','Mr','Mrs','Mrs','Other','Other','Other','Mr','Mr','Mr'], inplace = True)
df.loc[(df.Age.isnull())&(df.Title=='Mr'),'Age']= df.Age[df.Title=="Mr"].mean()



df.loc[(df.Age.isnull())&(df.Title=='Mrs'),'Age']= df.Age[df.Title=="Mrs"].mean()



df.loc[(df.Age.isnull())&(df.Title=='Master'),'Age']= df.Age[df.Title=="Master"].mean()



df.loc[(df.Age.isnull())&(df.Title=='Miss'),'Age']= df.Age[df.Title=="Miss"].mean()



df.loc[(df.Age.isnull())&(df.Title=='Other'),'Age']= df.Age[df.Title=="Other"].mean()



df = df.drop('Name', axis=1)



df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode().iloc[0])



df['Fare'] = df['Fare'].fillna(df['Fare'].mean())
df['Sex'] = df['Sex'].map( {'female': 1, 'male': 0} ).astype(int)

df['Embarked'] = df['Embarked'].map( {'Q': 0, 'S': 1, 'C': 2} ).astype(int)

df= df.drop(['Ticket', 'Cabin'], axis=1)

df['Title'] = pd.Categorical(df['Title'])

df['Title'] = df['Title'].cat.codes
df_train = df[df['train'] == 1]

df_test = df[df['train'] == 0]

df_train.drop('train', axis=1, inplace=True)

df_test.drop('train', axis=1, inplace=True)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

df_train['Survived'] = survived

X = df_train.drop(columns=["Survived"]).values

train_x = scaler.fit_transform(X)

train_y = df_train["Survived"].values

test = scaler.fit_transform(df_test.values)
from sklearn.model_selection import RandomizedSearchCV



# Number of trees in random forest

n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]

# Number of features to consider at every split

max_features = ['auto', 'sqrt']

# Maximum number of levels in tree

max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]

max_depth.append(None)

# Minimum number of samples required to split a node

min_samples_split = [2, 5, 10]

# Minimum number of samples required at each leaf node

min_samples_leaf = [1, 2, 4]

# Method of selecting samples for training each tree

bootstrap = [True, False]



# Create the grid

random_grid = {'n_estimators': n_estimators,

               'max_features': max_features,

               'max_depth': max_depth,

               'min_samples_split': min_samples_split,

               'min_samples_leaf': min_samples_leaf,

               'bootstrap': bootstrap}
from sklearn.ensemble import RandomForestClassifier



# First create the base model to tune

rf = RandomForestClassifier()



# Random search of parameters, using 5 fold cross validation, 

# search across 100 different combinations, and use all available cores

rf_clf = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 5, 

                            random_state=42, n_jobs = -1)



rf_clf.fit(train_x, train_y)
clf = rf_clf.best_estimator_
preds_rf = clf.predict(test)
rf_submit = pd.DataFrame({"Survived": preds_rf,"PassengerId": df_test.PassengerId})
rf_submit = rf_submit[["PassengerId","Survived"]]
rf_submit.to_csv("submission_rf.csv",index=False)
from xgboost import XGBClassifier

from sklearn.model_selection import GridSearchCV



xgb_model = XGBClassifier()



param_grid = {'n_estimators':[240,280,320],

              'max_depth':[10,11,12],

              'gamma':[0,1,2,3],

              'max_delta_step':[0,1,2],

              'min_child_weight':[1,2,3], 

              'colsample_bytree':[0.55,0.6,0.65],

              'learning_rate':[0.1,0.2,0.3]

            }

xgb_clf = GridSearchCV(xgb_model, param_grid = param_grid, scoring = 'neg_log_loss', error_score = 0,  n_jobs = -1)



xgb_clf.fit(train_x, train_y)
preds_xgb = xgb_clf.predict(test)
xgb_submit = pd.DataFrame({"Survived": preds_xgb,"PassengerId": df_test.PassengerId})

xgb_submit = xgb_submit[["PassengerId","Survived"]]

xgb_submit.to_csv("submission_xgb.csv",index=False)
from sklearn.ensemble import VotingClassifier



def get_voting(models):

    ensemble = VotingClassifier(estimators=models, voting='soft')

    return ensemble
models = [('xgb', xgb_clf.best_estimator_),

          ('rf', rf_clf.best_estimator_)]
voting_ens_model = get_voting(models)

voting_ens_model.fit(train_x, train_y)

voting_preds = voting_ens_model.predict(test)

voting_submit = pd.DataFrame({"Survived": voting_preds,"PassengerId": df_test.PassengerId})

voting_submit = voting_submit[["PassengerId","Survived"]]

voting_submit.to_csv("submission_ensembled.csv",index=False)
df_test.shape