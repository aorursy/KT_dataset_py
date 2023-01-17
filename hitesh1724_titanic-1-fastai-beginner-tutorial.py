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
!pip install fastai2
!pip install fastcore==0.1.35 ##Currently supported with fastai2
import fastcore

fastcore.__version__
import fastai2

fastai2.__version__
from fastai2.tabular.all import *
df_test= pd.read_csv('/kaggle/input/titanic-extended/test.csv')

df_train= pd.read_csv('../input/titanic-extended/train.csv')

df_train.head()
df_train.describe()
df_train.isnull().sum().sort_index()/len(df_train)
df_train.dtypes

g_train =df_train.columns.to_series().groupby(df_train.dtypes).groups

g_train
cat_names= [

        'Name', 'Sex', 'Ticket', 'Cabin', 

        'Embarked', 'Name_wiki', 'Hometown', 

        'Boarded', 'Destination', 'Lifeboat', 

        'Body'

]



cont_names = [ 

    'PassengerId', 'Pclass', 'SibSp', 'Parch', 

    'Age', 'Fare', 'WikiId', 'Age_wiki','Class'

 ]



splits = RandomSplitter(valid_pct=0.2)(range_of(df_train))



to = TabularPandas(df_train, procs=[Categorify, FillMissing,Normalize],

                   cat_names = cat_names,

                   cont_names = cont_names,

                   y_names='Survived',

                   splits=splits)
#df_train.dtypes

g_train =to.train.xs.columns.to_series().groupby(to.train.xs.dtypes).groups

g_train
to.train.xs.Age_na.head()
to.train
to.train.xs
to.train.ys.values.ravel()
### RANDOM FOREST

from sklearn.ensemble import RandomForestClassifier



X_train, y_train = to.train.xs, to.train.ys.values.ravel()

X_valid, y_valid = to.valid.xs, to.valid.ys.values.ravel()
X_train.head()
rnf_classifier= RandomForestClassifier(n_estimators=100, n_jobs=-1)

rnf_classifier.fit(X_train,y_train)
y_pred=rnf_classifier.predict(X_valid)





from sklearn.metrics import accuracy_score



accuracy_score(y_pred, y_valid)
df_test.head()
df_test.dtypes

g_train =df_test.columns.to_series().groupby(df_test.dtypes).groups

g_train
cat_names= [

        'Name', 'Sex', 'Ticket', 'Cabin', 

        'Embarked', 'Name_wiki', 'Hometown', 

        'Boarded', 'Destination', 'Lifeboat', 

        'Body'

]



cont_names = [ 

    'PassengerId', 'Pclass', 'SibSp', 'Parch', 

    'Age', 'Fare', 'WikiId', 'Age_wiki','Class'

 ]
test = TabularPandas(df_test, procs=[Categorify, FillMissing,Normalize],

                   cat_names = cat_names,

                   cont_names = cont_names,

                   )
X_test= test.train.xs
X_test.head()
X_test.dtypes

g_train =X_test.columns.to_series().groupby(X_test.dtypes).groups

g_train
X_test= X_test.drop('Fare_na', axis=1)
y_pred=rnf_classifier.predict(X_test)

y_pred= y_pred.astype(int)
output= pd.DataFrame({'PassengerId':df_test.PassengerId, 'Survived': y_pred})

output.to_csv('my_submission_titanic.csv', index=False)

output.head()