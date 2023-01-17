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

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
# load data

df_train = pd.read_csv("../input/titanic/train.csv")

df_test = pd.read_csv("../input/titanic/test.csv")
df_train.columns, df_test.columns
df_train.isnull().sum()
df_test.isnull().sum()
# drop catagorical columns

df_train.drop(columns = ["Name" ,"Ticket" , "Cabin"], inplace=True)
df_test.drop(columns = ["Name" ,"Ticket" , "Cabin"], inplace=True)
# missing values

df_train.Age.fillna(df_train.Age.mean() , inplace= True)

df_test.Age.fillna(df_train.Age.mean() , inplace = True)
df_train.isnull().sum()
df_train.Embarked.fillna(df_train.Embarked.mode()[0], inplace = True)

df_test.Embarked.fillna(df_test.Embarked.mode()[0] , inplace = True)
df_test.isnull().sum()
df_test.Fare.fillna(df_train.Fare.mean() , inplace = True)

df_test.isnull().sum()
df_train.Sex.replace("male" , 0 , inplace =True)

df_train.Sex.replace("female" , 1 , inplace =True)

df_train.Embarked.replace("C" , 0 , inplace =True)

df_train.Embarked.replace("S" , 1 , inplace =True)

df_train.Embarked.replace("Q" , 2 , inplace =True)
df_test.Sex.replace("male" , 0 , inplace =True)

df_test.Sex.replace("female" , 1 , inplace =True)

df_test.Embarked.replace("C" , 0 , inplace =True)

df_test.Embarked.replace("S" , 1 , inplace =True)

df_test.Embarked.replace("Q" , 2 , inplace =True)
# Build Model

X = df_train.drop(columns = ["Survived"])

y = df_train["Survived"]
from sklearn.model_selection import train_test_split

train_X, val_X, train_y, val_y = train_test_split(X, y, test_size=0.2, random_state=0)
from sklearn.tree import DecisionTreeClassifier

classifier = DecisionTreeClassifier(max_leaf_nodes=8 , random_state = 1)

classifier.fit(train_X,train_y)

preds_val = classifier.predict(df_test)
test_out = pd.DataFrame({

    'PassengerId': df_test.PassengerId, 

    'Survived': preds_val

})

test_out.to_csv('submission.csv', index=False)