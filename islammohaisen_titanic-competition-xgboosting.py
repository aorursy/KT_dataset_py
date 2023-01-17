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
# load data

train_df = pd.read_csv("../input/titanic/train.csv", index_col = 'PassengerId')

test_df = pd.read_csv("../input/titanic/test.csv")
index = test_df['PassengerId']
test_df = pd.read_csv("../input/titanic/test.csv",  index_col = 'PassengerId')
train_test_data = [train_df, test_df]



for dataset in train_test_data:

    dataset['Title'] = dataset['Name'].str.extract('([A-Za-z]+)\.', expand =False)
train_df['Title'].value_counts()
test_df['Title'].value_counts()
# title mapping

title_mapping = {'Mr':0 , 'Miss':1, 'Mrs':2, 'Master':3,'Dr':3 , 'Rev':3 , 'Col':3, 'Mlle':3 ,

                 'Major':3, 'Don':3, 'Sir':3,'Ms':3,'Jonkheer':3, 'Capt':3 , 'Lady':3,

                 'Countess':3 , 'Mme':3 }



for dataset in train_test_data:

    dataset['Title'] = dataset['Title'].map(title_mapping)
train_df.head()
test_df.head()
train_df.drop('Name', axis = 1 , inplace = True)

test_df.drop('Name', axis = 1 , inplace = True)
test_df['Title'].isnull()
test_df['Title'].isnull()
train_df.head()
test_df.head()
# sex mapping

# title mapping

sex_mapping = {'male':0, 'female':1}



for dataset in train_test_data:

    dataset['Sex'] = dataset['Sex'].map(sex_mapping)
train_df.head()
train_df['Age'].fillna(train_df.groupby('Title')['Age'].transform('median'), inplace =True)

test_df['Age'].fillna(test_df.groupby('Title')['Age'].transform('median'), inplace =True)
for dataset in train_test_data:

    dataset.loc[dataset['Age']<= 16, 'Age'] =0,

    dataset.loc[dataset['Age']> 16 & (dataset['Age']<=26), 'Age'] =1,

    dataset.loc[dataset['Age']> 26 & (dataset['Age']<=36), 'Age'] =2,

    dataset.loc[dataset['Age']> 36 & (dataset['Age']<=62), 'Age'] =3,

    dataset.loc[dataset['Age']> 62, 'Age'] =4
train_df
# Embarked

for dataset in train_test_data:

    dataset['Embarked'] = dataset['Embarked'].fillna('S')
embarked_mapping = {'S':0, 'C':1, 'Q':2}

for dataset in train_test_data:

    dataset['Embarked'] = dataset['Embarked'].map(embarked_mapping)
train_df['Fare'].fillna(train_df.groupby('Pclass')['Fare'].transform('median'), inplace = True)

test_df['Fare'].fillna(test_df.groupby('Pclass')['Fare'].transform('median'), inplace = True)
for dataset in train_test_data:

    dataset.loc[dataset['Fare']<= 17, 'Fare'] =0,

    dataset.loc[dataset['Fare']> 17 & (dataset['Fare']<=30), 'Fare'] =1,

    dataset.loc[dataset['Fare']> 30 & (dataset['Fare']<=100), 'Fare'] =2,

    dataset.loc[dataset['Fare']> 100, 'Fare'] =4
for dataset in train_test_data:

    dataset['Cabin'] = dataset['Cabin'].astype(str).str[:1]
cabin_mapping = {'A':0, 'B':0.4, 'C': 0.8, 'D':1.2, 'E':1.6, 'F':2, 'G':2.4, 'T':2.8}



for dataset in train_test_data:

    dataset['Cabin'] = dataset['Cabin'].map(cabin_mapping)
train_df['Cabin'].fillna(train_df.groupby('Pclass')['Cabin'].transform('median'), inplace = True)

test_df['Cabin'].fillna(test_df.groupby('Pclass')['Cabin'].transform('median'), inplace = True)
train_df['FamilySize'] = train_df['SibSp'] + train_df['Parch'] + 1

test_df['FamilySize'] = test_df['SibSp'] + test_df['Parch'] + 1
family_mapping = {1:0, 2:0.4, 3:0.8, 4:1.2, 5:1.6, 6:2, 7:2.4, 8:2.8, 9:3.2, 10:3.6, 11:4}



for dataset in train_test_data:

    dataset['FamilySize'] = dataset['FamilySize'].map(family_mapping)
feature_drop = ['Ticket', 'SibSp', 'Parch']

train_df = train_df.drop(feature_drop, axis = 1)

test_df = test_df.drop(feature_drop, axis =1)
test_df.describe()
train_df.describe()
test_df['Title'].isnull()
test_df.describe()
x = train_df.drop(["Survived"] , axis =1)

y = train_df["Survived"]
from sklearn.model_selection import train_test_split

x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=0)
test_df.shape
test_df.isna().sum()
test_df['Title'].isna().index
test_df['Title'].isnull()
test_df.loc[1306,'Title'] = 0
test_df['Title'].isnull()
train_df.isna().sum()
from xgboost import XGBClassifier

my_model = XGBClassifier(n_estimators=200, learning_rate=0.05, max_depth =6)

my_model.fit(x_train, y_train)

predictions = my_model.predict(test_df)
test_out = pd.DataFrame({

    'PassengerId': index, 

    'Survived': predictions

})

test_out.to_csv('submission.csv', index=False)