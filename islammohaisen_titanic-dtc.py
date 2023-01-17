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
import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
# load data

df_train = pd.read_csv("../input/titanic/train.csv")

df_test = pd.read_csv("../input/titanic/test.csv")
df_train.shape, df_test.shape
df_train.drop(columns = ["Name", 'Ticket'], inplace=True)

df_test.drop(columns = ["Name",'Ticket'], inplace=True)
df_train.shape, df_test.shape
df_train.head()
df_test.head()
df_train.dtypes
df_test.dtypes
df_train.isnull().sum()
df_test.isnull().sum()
df_train.Embarked.fillna('Missing', inplace = True)
df_train.Cabin.fillna('Missing', inplace = True)

df_test.Cabin.fillna('Missing' , inplace = True)
df_train.Cabin = df_train.Cabin.astype(str).str[0]

df_test.Cabin = df_test.Cabin.astype(str).str[0]
df_train.head()
df_train.head()
df_train.Age.fillna(df_train.Age.median() , inplace= True)

df_test.Age.fillna(df_train.Age.median() , inplace = True)
df_test.Fare.fillna(df_train.Fare.median() , inplace = True)
df_train.head()
df_train_temp = df_train.copy()

df_train_frequency_map= df_train_temp['Sex'].value_counts().to_dict()

df_train_temp['Sex']= df_train_temp['Sex'].map(df_train_frequency_map)



df_train_frequency_map1= df_train_temp['Cabin'].value_counts().to_dict()

df_train_temp['Cabin']= df_train_temp['Cabin'].map(df_train_frequency_map1)



df_train_frequency_map2= df_train_temp['Embarked'].value_counts().to_dict()

df_train_temp['Embarked']= df_train_temp['Embarked'].map(df_train_frequency_map2)
df_train_temp.head()
df_test_temp = df_test.copy()



df_test_temp['Sex']= df_test_temp['Sex'].map(df_train_frequency_map)

df_test_temp['Cabin']= df_test_temp['Cabin'].map(df_train_frequency_map1)

df_test_temp['Embarked']= df_test_temp['Embarked'].map(df_train_frequency_map2)
df_test_temp.head()
X = df_train_temp.drop(columns = ["Survived"])

y = df_train_temp["Survived"]
# data splitting

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
X_train.shape, X_test.shape, y_train.shape, y_test.shape
from xgboost import XGBClassifier

booster = XGBClassifier(random_state =44, max_depth = 2)

booster.fit(X_train, y_train)

preds = booster.predict(df_test_temp

                    )
test_out = pd.DataFrame({

    'PassengerId': df_test_temp.PassengerId, 

    'Survived': preds

})

test_out.to_csv('submission.csv', index=False)