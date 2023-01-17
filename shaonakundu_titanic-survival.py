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
train_data = pd.read_csv("/kaggle/input/titanic/train.csv")

train_data.head()
test_data = pd.read_csv("/kaggle/input/titanic/test.csv")

test_data.head()
women = train_data.loc[train_data.Sex=='female']['Survived']

rate_women = sum(women)/len(women)

print("% women survived: ",rate_women)
men = train_data.loc[train_data.Sex=='male']['Survived']

rate_men = sum(men)/len(men)

print("% men survived: ",rate_men)
total = train_data.isnull().sum().sort_values(ascending = False)

percent_1 = train_data.isnull().sum()/train_data.isnull().count()*100

percent_2 = (round(percent_1, 1)).sort_values(ascending=False)

missing_data = pd.concat([total, percent_2], axis=1, keys=['Total', '%'])

missing_data.head(5)

train_data.columns
train_data = train_data.drop(['PassengerId'], axis =1)

train_data = train_data.drop(['Cabin'], axis=1)

test_data = test_data.drop(['Cabin'], axis=1)
data = [train_data, test_data]



for dataset in data:

    mean = train_data["Age"].mean()

    std = test_data["Age"].std()

    is_null = dataset["Age"].isnull().sum()

    rand_age = np.random.randint(mean - std, mean + std, size = is_null)

    age_slice = dataset["Age"].copy()

    age_slice[np.isnan(age_slice)] = rand_age

    dataset["Age"] = age_slice

    dataset["Age"] = train_data["Age"].astype(int)

train_data["Age"].isnull().sum()
train_data["Embarked"].describe()
common_value = 'S'

data = [train_data, test_data]

for dataset in data:

    dataset["Embarked"] = dataset["Embarked"].fillna(common_value)
train_data.info()
data = [train_data, test_data]



for dataset in data:

    dataset['Fare'] = dataset['Fare'].fillna(0)

    dataset['Fare'] = dataset['Fare'].astype(int)
train_data = train_data.drop(['Name'], axis=1)

test_data = test_data.drop(['Name'], axis=1)
genders = {"male": 0, "female": 1}

data = [train_data, test_data]



for dataset in data:

    dataset['Sex'] = dataset['Sex'].map(genders)
train_data['Ticket'].describe()
train_data = train_data.drop(['Ticket'], axis=1)

test_data = test_data.drop(['Ticket'], axis=1)
ports = {"S": 0, "C": 1, "Q": 2}

data = [train_data, test_data]



for dataset in data:

    dataset['Embarked'] = dataset['Embarked'].map(ports)
train_data.head(8)
import sklearn

from sklearn.ensemble import RandomForestClassifier



from sklearn import metrics 


x = train_data.drop("Survived", axis=1)

y = train_data["Survived"]

x_test = test_data.drop("PassengerId", axis=1).copy()



ran_forest = RandomForestClassifier(n_estimators = 100)

ran_forest.fit(x,y)

y_pred = ran_forest.predict(x_test)

ran_forest.score(x,y)
from sklearn.model_selection import cross_val_predict

from sklearn.metrics import confusion_matrix

pred = cross_val_predict(ran_forest, x, y, cv=3)

confusion_matrix(y, pred)
from sklearn.metrics import precision_score, recall_score

print("precision: ",precision_score(y, pred))

print("recall: ",recall_score(y, pred))
from sklearn.metrics import f1_score

f1_score(y,pred)
output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': y_pred})

output.to_csv('my_submission.csv', index=False)

print("Your submission was successfully saved!")