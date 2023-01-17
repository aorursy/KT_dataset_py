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
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

titanic_file_path_train = '/kaggle/input/titanic/train.csv'
titanic_file_path_test = '/kaggle/input/titanic/test.csv'

dt_train = pd.read_csv(titanic_file_path_train)

dt_test = pd.read_csv(titanic_file_path_test)

dt_train.info()

cols_to_drop = ['PassengerId','Name','Ticket','Cabin','Embarked']
data_clean = dt_train.drop(columns=cols_to_drop,axis=1)
data_clean_test = dt_test.drop(columns=cols_to_drop,axis=1)
data_clean.info()

# now I can clean using LabelEncoder pre-processing
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data_clean['Sex'] = le.fit_transform(data_clean['Sex'])
data_clean_test['Sex'] = le.fit_transform(data_clean_test['Sex'])

data_clean['Age'] = data_clean.fillna(data_clean['Age'].mean())['Age']
data_clean_test['Age'] = data_clean_test.fillna(data_clean_test['Age'].mean())['Age']
data_clean_test['Fare'] = data_clean_test.fillna(data_clean_test['Fare'].mean())['Fare']
input_cols = ['Pclass',"Sex","Age","SibSp","Parch","Fare"]
output_cols = ["Survived"]

x_train = data_clean[input_cols]
y_train = data_clean[output_cols]
x_test = data_clean_test[input_cols]

from sklearn.tree import DecisionTreeClassifier

dtc = DecisionTreeClassifier(max_depth=4,criterion='entropy')
dtc.fit(x_train,y_train)
dtc.score(x_train,y_train)