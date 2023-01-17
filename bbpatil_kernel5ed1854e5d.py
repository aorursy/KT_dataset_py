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
df1 = pd.read_csv('../input/titanic/train.csv')

df2 = pd.read_csv('../input/titanic/test.csv')

df = pd.concat([df1,df2],axis=0)
df.info()
df['Age']=df['Age'].fillna(df['Age'].mean())

df['Embarked']= df['Embarked'].fillna(df['Embarked'].mode()[0])
df.drop(['Cabin','Name','PassengerId','Ticket'],axis=1,inplace=True)
df.info()
df.head()
df.head()
df3 = pd.get_dummies(df,columns=['Sex','Embarked'])
df3.head()
df3.shape
df3.info()
df4 = df3.iloc[: 891 , : ]

df5 = df3.iloc[891 : , : ]
df5.head()
df5.drop(['Survived'],axis=1,inplace=True)
df5.head()
X = df4.drop(['Survived'],axis=1)

y = df4['Survived']
X.head()
import xgboost

regressor = xgboost.XGBRegressor()

booster=['gbtree','gblinear']

base_score=[0.25,0.5,0.75,1]
n_estimators = [100, 500, 900, 1100, 1500]

max_depth = [2, 3, 5, 10, 15]

booster=['gbtree','gblinear']

learning_rate=[0.05,0.1,0.15,0.20]

min_child_weight=[1,2,3,4]



# Define the grid of hyperparameters to search

hyperparameter_grid = {

    'n_estimators': n_estimators,

    'max_depth':max_depth,

    'learning_rate':learning_rate,

    'min_child_weight':min_child_weight,

    'booster':booster,

    'base_score':base_score

}
from sklearn.model_selection import RandomizedSearchCV

random_cv = RandomizedSearchCV(estimator=regressor,

            param_distributions=hyperparameter_grid,

            cv=5, n_iter=50,

            scoring = 'neg_mean_absolute_error',n_jobs = 4,

            verbose = 5, 

            return_train_score = True,

            random_state=42)

random_cv.fit(X,y)
random_cv.best_estimator_
regressor = xgboost.XGBRegressor(base_score=1, booster='gbtree', colsample_bylevel=1,

             colsample_bynode=1, colsample_bytree=1, gamma=0,

             importance_type='gain', learning_rate=0.1, max_delta_step=0,

             max_depth=15, min_child_weight=1, missing=None, n_estimators=900,

             n_jobs=1, nthread=None, objective='reg:linear', random_state=0,

             reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,

             silent=None, subsample=1, verbosity=1)
regressor.fit(X,y)
y_pred=regressor.predict(df5).astype(int)
y_pred
pred=pd.DataFrame(y_pred)

sub_df=pd.read_csv('../input/titanic/gender_submission.csv')

sub_df['Survived'] = pred

sub_df.to_csv('submission.csv',index=False)