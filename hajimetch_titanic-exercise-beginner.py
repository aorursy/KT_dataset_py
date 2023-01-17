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
import pandas as pd

df      = pd.read_csv('/kaggle/input/titanic/train.csv')

df_test = pd.read_csv('/kaggle/input/titanic/test.csv')

print(df)
df.mean()
df.mode()
df = df.fillna({'Age': 29.7, 'Fare': 32.2, 'Cabin': 'B96', 'Embarked': 'S'})

print(df.isnull().any())
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.compose import make_column_transformer

onehotencoding = ['Sex', 'Cabin', 'Embarked', 'Ticket']

standardscaling = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']

preprocessor = make_column_transformer((OneHotEncoder(categories='auto', handle_unknown='ignore'), onehotencoding),

                                       (StandardScaler(), standardscaling),

                                       remainder='drop')
preprocessor.fit(df.drop('Survived', axis=1))

X = preprocessor.transform(df.drop('Survived', axis=1)).toarray()

y = df['Survived'].values
from sklearn.ensemble import RandomForestClassifier

forest = RandomForestClassifier(criterion='gini', n_estimators=10000, random_state=1, n_jobs=-1)

forest.fit(X,y)
print(df_test)
df_test = df_test.fillna({'Age': 29.7, 'Fare': 32.2, 'Cabin': 'B96', 'Embarked': 'S'})

X_test = preprocessor.transform(df_test).toarray()
y_test = forest.predict(X_test)
df_sample = pd.read_csv('/kaggle/input/titanic/gender_submission.csv')

print(df_sample)
df_submit = pd.DataFrame({'PassengerId': df_test['PassengerId'], 'Survived': y_test})

print(df_submit)
df_submit.to_csv("/kaggle/working/submission.csv", index=False)