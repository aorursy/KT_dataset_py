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
train_df = pd.read_csv('/kaggle/input/titanic/train.csv')

train_df.head()
test_df = pd.read_csv('/kaggle/input/titanic/test.csv')
train_df.drop(['PassengerId','Name','Ticket','Cabin'],axis=1, inplace=True)
train_df.head()
train_df.isna().sum()
train_df.replace(np.nan, train_df.mean(), inplace=True)
from sklearn.preprocessing import LabelEncoder



col = train_df.drop(train_df.select_dtypes(exclude=['object']), axis=1).columns

print(col)



en1 = LabelEncoder()

train_df[col[0]] = en1.fit_transform(train_df[col[0]].astype('str'))



en2 = LabelEncoder()

train_df[col[1]] = en2.fit_transform(train_df[col[1]].astype('str'))
train_df.head()
X = train_df.iloc[:, 1:].values

Y = train_df.iloc[:, 0].values
print(X)
print(Y)
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0)
from sklearn.ensemble import RandomForestClassifier

classifier_forest = RandomForestClassifier()

classifier_forest.fit(X_train, Y_train)
from sklearn.metrics import confusion_matrix, accuracy_score

pred_forest = classifier_forest.predict(X_test)

cm = confusion_matrix(Y_test, pred_forest)

print(cm)

accuracy_score(Y_test, pred_forest)
test_df.head()
test_df.isna().sum()
test_df = test_df.drop(["Cabin"], axis=1)



test_df['Age'].fillna(test_df['Age'].median(), inplace=True)

test_df['Fare'] = test_df['Fare'].ffill()
test_df.info()
PassengerId = test_df['PassengerId']

test_df = test_df.drop(['PassengerId', 'Ticket', 'Name'], axis = 1) 
test_df[col[0]] = en1.transform(test_df[col[0]].astype('str'))



test_df[col[1]] = en2.transform(test_df[col[1]].astype('str'))
test_df.head()
y_pred_test = classifier_forest.predict(test_df)
submission = pd.DataFrame({

    'PassengerId' : PassengerId,

    'Survived' : y_pred_test

})



submission.to_csv('./submission.csv')