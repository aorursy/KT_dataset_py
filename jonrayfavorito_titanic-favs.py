# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
!ls /kaggle/input/titanic
train = pd.read_csv('/kaggle/input/titanic/train.csv')

test = pd.read_csv('/kaggle/input/titanic/test.csv')



train['Age'].fillna(train['Age'].mean(),inplace=True)

test['Age'].fillna(test['Age'].mean(),inplace=True)

test['Fare'].fillna(test['Fare'].mean(),inplace=True)

train['Embarked'].fillna(value='S', inplace=True)

train['family']=train['SibSp']+train['Parch']+1

test['family']=test['SibSp']+train['Parch']+1

train['Title'] = train['Name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]

test['Title'] = test['Name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]



train['FareBin'] = pd.qcut(train['Fare'], 4)

test['FareBin'] = pd.qcut(test['Fare'], 4)

train['AgeBin'] = pd.qcut(train['Age'], 5)

test['AgeBin'] = pd.qcut(test['Age'], 5)



from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler, MinMaxScaler

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import OneHotEncoder

from sklearn.compose import ColumnTransformer

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import GridSearchCV



X_train=train.drop(columns=['Survived','Cabin','Fare','Age','PassengerId','Ticket','SibSp','Parch','Name'])

Y_train=train.Survived

X_test=test.drop(columns=['Cabin','Fare','Age','PassengerId','Ticket','SibSp','Parch','Name'])



num_feat=X_train.select_dtypes(include='number').columns.to_list()

cat_feat=X_train.select_dtypes(include='object').columns.to_list()



num_pipe=Pipeline([

    ('imputer', SimpleImputer(strategy='mean')),

    ('scale', MinMaxScaler())

])



cat_pipe=Pipeline([

    ('imputer', SimpleImputer(strategy='most_frequent')),

    ('coder', OneHotEncoder(handle_unknown='ignore'))

])



ct=ColumnTransformer(remainder='drop',

    transformers=[

    ('numerical',num_pipe, num_feat),

    ('categorical',cat_pipe, cat_feat)

])



model_new=Pipeline([

    ('transformer', ct),

    ('predictor', RandomForestClassifier())

])



grid_values = {

     'predictor__bootstrap': [False, True],

     'predictor__n_estimators': [10, 30, 80,90, 100, 110, 130],

     'predictor__max_features': [0.6, 0.65, 0.7, 0.73, 0.7500000000000001, 0.78, 0.8],

     'predictor__min_samples_leaf': [10, 12, 14],

     'predictor__min_samples_split': [3, 5, 7],

     'predictor__criterion' : ['gini', 'entropy']

}



gcf = GridSearchCV(model_new, param_grid=grid_values)

gcf.fit(X_train, Y_train)

# model_new.fit(X_train, Y_train);

gender_submission=pd.read_csv('/kaggle/input/titanic/gender_submission.csv') 

results = pd.DataFrame({'PassengerId' : gender_submission['PassengerId'], 'Survived' : gcf.predict(X_test)})

results.to_csv('submission_title_n2.csv', index=False) # Submission csv file
gcf.best_params_