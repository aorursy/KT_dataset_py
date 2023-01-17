import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt
train=pd.read_csv("../input/titanic/train.csv")

test=pd.read_csv("../input/titanic/test.csv")

p_id=test['PassengerId']
train.head()
train.info()
test.info()
test["Fare"].fillna(test['Fare'].median(),inplace=True)
train.describe()
sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')
train.drop('Cabin',axis=1,inplace=True)

test.drop('Cabin',axis=1,inplace=True)
plt.figure(figsize=(12, 7))

sns.boxplot(x='Pclass',y='Age',data=train,palette='winter')
def impute_age_tr(cols):

    age_fill=train[['Age','Pclass']].groupby('Pclass').mean()

    Age = cols[0]

    Pclass = cols[1]

    

    if pd.isnull(Age):

        return int(age_fill['Age'][Pclass])

    else:

        return Age

def impute_age_te(cols):

    age_fill=test[['Age','Pclass']].groupby('Pclass').mean()

    Age = cols[0]

    Pclass = cols[1]

    

    if pd.isnull(Age):

        return int(age_fill['Age'][Pclass])

    else:

        return Age
train['Age'] = train[['Age','Pclass']].apply(impute_age_tr,axis=1)

test['Age'] = test[['Age','Pclass']].apply(impute_age_te,axis=1)
train['Embarked'].fillna("S",inplace=True)



test['Embarked'].fillna("S",inplace=True)
sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')
def sexx(Sex):

    if Sex=="male":

        return 0

    else:

        return 1

    

def age_new(age):

    if age<=16:

        return 0

    elif age>16 and age<=32:

        return 1

    elif age>32 and age<=48:

        return 2

    elif age>48 and age<=64:

        return 3

    else:

        return 4

    

def fare_new(fare):

    if fare<=7.91:

        return 0

    elif fare>7.91 and fare<=14.454:

        return 1

    elif fare>14.454 and fare<=31:

        return 2

    elif fare>31:

        return 3
train['Sex'] = train['Sex'].apply(sexx)

train['Age'] = train['Age'].apply(age_new)

train['Fare'] = train['Fare'].apply(fare_new)



test['Sex'] = test['Sex'].apply(sexx)

test['Age'] = test['Age'].apply(age_new)

test['Fare'] = test['Fare'].apply(fare_new)
train.drop(['PassengerId','Name','Embarked','Ticket'],axis=1,inplace=True)

test.drop(['PassengerId','Name','Embarked','Ticket'],axis=1,inplace=True)
train.head()
corrmat=train.corr()

plt.figure(figsize=(15, 10))

sns.heatmap(train[corrmat.index].corr(),annot=True,cmap="RdYlGn")
sns.distplot(train['Age'])
sns.distplot(train['Fare'])
y=train['Survived'].values
train.drop('Survived',axis=1,inplace=True)
X=train.values

X_test=test.values
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)

model.fit(X, y)

predictions = model.predict(X_test)
pred={

    'PassengerId':p_id,

    'Survived':predictions

}
df=pd.DataFrame(pred)
df.to_csv('./rf_pred.csv', index=False)
from sklearn.linear_model import LogisticRegression



logmodel = LogisticRegression()

logmodel.fit(X,y)

predictions_l = logmodel.predict(X_test)



pred_l={

    'PassengerId':p_id,

    'Survived':predictions_l

}



df_l=pd.DataFrame(pred_l)



df_l.to_csv('log_pred.csv', index=False)
## Hyperparameter optimization using RandomizedSearchCV

from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

import xgboost
## Hyper Parameter Optimization



params={

 "learning_rate"    : [0.05, 0.10, 0.15, 0.20, 0.25, 0.30 ] ,

 "max_depth"        : [ 3, 4, 5, 6, 8, 10, 12, 15],

 "min_child_weight" : [ 1, 3, 5, 7 ],

 "gamma"            : [ 0.0, 0.1, 0.2 , 0.3, 0.4 ],

 "colsample_bytree" : [ 0.3, 0.4, 0.5 , 0.7 ]

    

}
def timer(start_time=None):

    if not start_time:

        start_time = datetime.now()

        return start_time

    elif start_time:

        thour, temp_sec = divmod((datetime.now() - start_time).total_seconds(), 3600)

        tmin, tsec = divmod(temp_sec, 60)

        print('\n Time taken: %i hours %i minutes and %s seconds.' % (thour, tmin, round(tsec, 2)))
classifier=xgboost.XGBClassifier()
random_search=RandomizedSearchCV(classifier,param_distributions=params,n_iter=5,scoring='roc_auc',n_jobs=-1,cv=5,verbose=3)
from datetime import datetime

# Here we go

start_time = timer(None) # timing starts from this point for "start_time" variable

random_search.fit(X,y)

timer(start_time) # timing ends here for "start_time" variable
random_search.best_estimator_
random_search.best_params_
classifier=xgboost.XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,

              colsample_bynode=1, colsample_bytree=0.4, gamma=0.0, gpu_id=-1,

              importance_type='gain', interaction_constraints='',

              learning_rate=0.3, max_delta_step=0, max_depth=5,

              min_child_weight=5, monotone_constraints='()',

              n_estimators=100, n_jobs=0, num_parallel_tree=1,

              objective='binary:logistic', random_state=0, reg_alpha=0,

              reg_lambda=1, scale_pos_weight=1, subsample=1,

              tree_method='exact', validate_parameters=1, verbosity=None)
classifier.fit(X,y)
classifier.score(X,y)
predictions_XG = classifier.predict(X_test)



pred_xg={

    'PassengerId':p_id,

    'Survived':predictions_XG

}



df_xg=pd.DataFrame(pred_xg)



df_xg.to_csv('xg_pred.csv', index=False)