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
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



train = pd.read_csv('../input/titanic/train.csv')

test  = pd.read_csv('../input/titanic/test.csv')

train.shape
train.isnull().sum()
X = train.drop('Survived',axis=1).copy()

y = train['Survived'].copy()

#X_train.shape

y.shape
from sklearn.model_selection import train_test_split

X_train,X_val,y_train, y_val= train_test_split(X,y,test_size=0.2,random_state=0)
X_test=test.copy()

X_test.shape
train.info()
num_col_missing=['Age']

cat_col_missing=['Embarked']

num_cols = [col for col in X_train.columns if X_train[col].dtype in ['int64', 'float64']]

cat_cols = [col for col in X_train.columns if X_train[col].dtype=='object']

print('numerical columns: ',num_cols)

print('Categorical Columns: ', cat_cols)
from sklearn.impute import SimpleImputer

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

#from xgboost import XGBClassifier

numerical_imputer = SimpleImputer(strategy='mean')

categorical_imputer = Pipeline(steps=[

    ('imputer',SimpleImputer(strategy='most_frequent')),

    ('encoder',OneHotEncoder(handle_unknown='ignore'))    

])



preprocessor= ColumnTransformer(transformers=[

    ('num',numerical_imputer,num_cols),

    ('cat',categorical_imputer,cat_cols)

])
from sklearn.ensemble import RandomForestClassifier

model= RandomForestClassifier(n_estimators=50, criterion='entropy', random_state=0)
my_pipeline = Pipeline(steps=[('preprocessor',preprocessor),

                             ('model',model)])
my_pipeline.fit(X_train, y_train)

preds = my_pipeline.predict(X_test)

preds.shape
actual_result = pd.read_csv('../input/titanic/gender_submission.csv')

actual_result
from sklearn.metrics import mean_absolute_error

error= mean_absolute_error(actual_result['Survived'],preds)

print(error)
Pred_result = pd.DataFrame({

        "PassengerId": X_test["PassengerId"],

        "Survived": preds})

Pred_result.to_csv('submission.csv', index=False)

submission = pd.read_csv('./submission.csv')

submission.head()