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
#Read Data

import pandas as pd

file_path = '../input/titanic/train.csv'

df = pd.read_csv(file_path, index_col='PassengerId')

df
# Missing Value

df.isna().sum()
df.Survived.value_counts()
len(df.Name.unique())
len(df.Ticket.unique())
len(df.Cabin.unique())
for col in df.columns:

    print(col, len(df[col].unique()))
# drop (Name, Ticket, Cabin)

df.drop(columns=['Name', 'Cabin', 'Ticket'], inplace=True)

df
#count missing value remaining

df.isna().sum()
# replace missing values in the Age column with mean

age_Mean = df.Age.mean()

df.Age.fillna(age_Mean, inplace=True)
df.isna().sum()
# find type of values in the Embarked column

df.Embarked.dtype
df.Embarked.value_counts()
# the most repetitive value in a column

df.Embarked.mode()
# replace missing values in the Embarked column with repetitive value 

df.Embarked.fillna('S', inplace=True)
# check missing values after replacement

df.isna().sum()
df
#We note that the gender column is categorized and not numbers, and machine learning prefers dealing with numerical values, so we will replace the values of this table with numbers

df.Embarked.replace('C', 0, inplace=True)

df.Embarked.replace('S', 1, inplace=True)

df.Embarked.replace('Q', 2, inplace=True)
df
import seaborn as sns

import matplotlib.pyplot as plt

plt.figure(figsize=(16,8))

sns.heatmap(df.corr())
df.Sex.replace('male', 0, inplace=True)

df.Sex.replace('female', 1, inplace=True)

df
y = df.Survived

X = df.drop(columns=['Survived'])

X
y
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y)
para = list(range(2, 15, 1))

para
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score, classification_report, f1_score

results = {}

for i in para:

    dt = DecisionTreeClassifier(max_leaf_nodes=i, random_state=1)

    dt.fit(X_train, y_train)

    preds = dt.predict(X_test)

    acc = accuracy_score(y_true=y_test, y_pred=preds)

    f1 = f1_score(y_true=y_test, y_pred=preds)

    print(i)

    print(classification_report(y_true=y_test, y_pred=preds))

    results[i] = f1
max(results, key=results.get)

results[max(results, key=results.get)]

best_para = max(results, key=results.get)

final_model = DecisionTreeClassifier(max_leaf_nodes=best_para)

final_model.fit(X, y)
testpath = '../input/titanic/test.csv'

test_df = pd.read_csv(testpath, index_col='PassengerId')

test_df.drop(columns=['Name', 'Cabin', 'Ticket'], inplace=True)

test_df
test_df.isna().sum()
age_mean = df.Age.mean()

df.Age.fillna(age_mean, inplace=True)

df.isna().sum()
test_df.Age.fillna(age_mean, inplace=True)

test_df.Fare.fillna(df.Fare.mean(), inplace=True)
test_df
test_df.Sex.replace('male', 0, inplace=True)

test_df.Sex.replace('female', 1, inplace=True)

test_df.Embarked.replace('C', 0, inplace=True)

test_df.Embarked.replace('S', 1, inplace=True)

test_df.Embarked.replace('Q', 2, inplace=True)

test_df
preds = final_model.predict(test_df)

preds.shape
test_df.shape
test_out = pd.DataFrame({

    'PassengerId': test_df.index, 

    'Survived': preds

})
test_out.to_csv('submission.csv', index=False)