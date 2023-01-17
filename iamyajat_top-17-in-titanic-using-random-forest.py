# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.impute import SimpleImputer

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_absolute_error



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
imputer = SimpleImputer()

test_data = pd.read_csv("../input/titanic/test.csv")

train_data = pd.read_csv("../input/titanic/train.csv")

train_data = train_data[train_data['Embarked'].notna()]

train_data['Embarked'].isnull().sum()

train_data.describe()
clean_data = train_data.drop(['PassengerId','Name','Cabin','Ticket','Survived'], axis=1)

clean_data.head()
clean_test_data = test_data.drop(['PassengerId','Name','Cabin','Ticket'], axis=1)

clean_test_data.head()

clean_test_data['Embarked'].isnull().sum()
sex = {'male': 1,'female': 2}

emb = {'S': 1,'C': 2, 'Q': 3, None:0}

clean_data.Sex = [sex[item] for item in clean_data.Sex]

clean_test_data.Sex = [sex[item] for item in clean_test_data.Sex]

clean_data.Embarked = [emb[item] for item in clean_data.Embarked]

clean_test_data.Embarked = [emb[item] for item in clean_test_data.Embarked]

clean_data.head()
y = train_data["Survived"]

X = imputer.fit_transform(clean_data)

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

X_test = imputer.fit_transform(clean_test_data)

A = np.zeros((10,10))



demo_model = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=1, criterion='entropy')

demo_model.fit(train_X, train_y)

val_predictions = demo_model.predict(val_X)

val_mae = mean_absolute_error(val_predictions, val_y)

print(val_mae)
model = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=1)

model.fit(X, y)
predictions = model.predict(X_test)
output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})

output.to_csv('submission.csv', index=False)

print("Your submission was successfully saved!")

output