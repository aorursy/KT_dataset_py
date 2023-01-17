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
from sklearn.tree import DecisionTreeClassifier

my_data = pd.read_csv('/kaggle/input/titanic/train.csv')

my_data.head(10)

my_data.count()
train_df = pd.read_csv('/kaggle/input/titanic/train.csv')

test_df = pd.read_csv('/kaggle/input/titanic/test.csv')
X = train_df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch']].values

X[0:5]
X_test = test_df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch']].values

X[0:5]
from sklearn import preprocessing

sex = preprocessing.LabelEncoder()

sex.fit(['female','male'])

X[:,1] = sex.transform(X[:,1]) 

X[0:5]

X_test[:,1] = sex.transform(X_test[:,1]) 

X_test[0:5]
y_train = train_df["Survived"]

y_train [0:5]
missing_val_count_by_column = (train_df.isnull().sum())

print(missing_val_count_by_column[missing_val_count_by_column > 0])
from sklearn.impute import SimpleImputer

my_imputer = SimpleImputer()

X_clean = my_imputer.fit_transform(X)

X_test_clean = my_imputer.fit_transform(X_test)
decision_tree = DecisionTreeClassifier()

decision_tree.fit(X_clean, y_train)

Y_pred = decision_tree.predict(X_test_clean)
acc_decision_tree = decision_tree.score(X_clean, y_train)

acc_decision_tree
test_df ["Survived"] = Y_pred

new = test_df [['PassengerId','Survived']]

new.to_csv('pred.csv',index=False)