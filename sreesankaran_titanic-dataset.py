# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd# data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.preprocessing import LabelEncoder

from sklearn.neural_network import MLPClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import train_test_split,GridSearchCV

from sklearn.metrics import confusion_matrix

from matplotlib import pyplot as plt

%matplotlib inline



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train_data = pd.read_csv("/kaggle/input/titanic/train.csv")

test_data = pd.read_csv("../input/titanic/test.csv")
train_data.info()
train_data['title'] = train_data.Name.str.extract("([A-z]+\.)")

test_data['title'] = test_data.Name.str.extract("([A-z]+\.)")
titles_train_group = train_data.title.value_counts(normalize=True)*100

train_data.title = train_data.title.replace(titles_train_group[titles_train_group<1].index,"Other")

titles_test_group = test_data.title.value_counts(normalize=True)*100

test_data.title = test_data.title.replace(titles_test_group[titles_test_group<1].index,"Other")
train_data['no_of_familymembrs'] = train_data['Parch']+train_data['SibSp']+1

test_data['no_of_familymembrs'] = test_data['Parch']+test_data['SibSp']+1
train_data.Cabin = train_data.Cabin.fillna('U')

train_data.Cabin = train_data.Cabin.str[0]

test_data.Cabin = test_data.Cabin.fillna('U')

test_data.Cabin = test_data.Cabin.str[0]
#Filling the age data with the average age

train_data.loc[train_data[train_data.Age.isnull()].index, "Age"] = train_data.Age.mean()

test_data.loc[test_data[test_data.Age.isnull()].index, "Age"] = test_data.Age.mean()
#binning the ages

train_data['age_by_decade'] = pd.cut(x=train_data['Age'], bins=[0,16, 20, 30, 40, 50, 60, 70, 80], labels=['child','youth','20s', '30s', '40s', '50s','60s','70s'])

test_data['age_by_decade'] = pd.cut(x=test_data['Age'], bins=[0,16, 20, 30, 40, 50, 60, 70, 80], labels=['child','youth','20s', '30s', '40s', '50s','60s','70s'])



survival_wrt_sex = train_data.groupby(['Sex', 'Survived']).PassengerId.count().reset_index([0,1])
survival_wrt_ticket_class = train_data.groupby(['Pclass', 'Survived']).PassengerId.count().reset_index([0,1])
survival_wrt_age = train_data.groupby(['age_by_decade', 'Survived', 'Sex']).PassengerId.count().reset_index([0,1,2])
train_data.Embarked.mode()
train_data.loc[train_data[train_data.Embarked.isnull()].index, "Embarked"] = 'S'

test_data.loc[test_data[test_data.Embarked.isnull()].index, "Embarked"] = 'S'
#Encoding the categorical values

le = LabelEncoder()

le.fit(train_data['Sex'])

train_data['sex_encoded'] = le.transform(train_data.Sex)

le.fit(train_data['age_by_decade'])

train_data['age_bin_encode'] = le.transform(train_data.age_by_decade)

le.fit(train_data.Cabin)

train_data.Cabin = le.transform(train_data.Cabin)

le.fit(train_data.title)

train_data.title = le.transform(train_data.title)

le.fit(train_data['Embarked'])

train_data['Embarked'] = le.transform(train_data.Embarked)

le.fit(test_data['Embarked'])

test_data['Embarked'] = le.transform(test_data.Embarked)

le.fit(test_data['Sex'])

test_data['sex_encoded'] = le.transform(test_data.Sex)

le.fit(test_data['age_by_decade'])

test_data['age_bin_encode'] = le.transform(test_data.age_by_decade)

le.fit(test_data.Cabin)

test_data.Cabin = le.transform(test_data.Cabin)

le.fit(test_data.title)

test_data.title = le.transform(test_data.title)
features = ['PassengerId','Survived', 'Pclass', 'sex_encoded', 'age_bin_encode', 'Cabin', 'no_of_familymembrs', 'title', 'Embarked']

data = train_data[features]

X_test = test_data[['Pclass', 'sex_encoded', 'age_bin_encode', 'Cabin', 'no_of_familymembrs', 'title', 'Embarked']].values
#spliting data into train and validation sets

training_set, validation_set = train_test_split(data, test_size = 0.2, random_state = 21)
X_val = validation_set.iloc[:,2:].values

y_val = validation_set.iloc[:, 1].values

X_train = training_set.iloc[:,2:].values

y_train  = training_set.iloc[:, 1].values
classifier = DecisionTreeClassifier(random_state=21, max_depth=3)
classifier.fit(X_train, y_train)
#validating model using validation set

y_pred = classifier.predict(X_val)
#confusion matrix to check accuracy

cm = confusion_matrix(y_pred, y_val)
# Function used to tell the accuracy. Credit AIM website.

def accuracy(confusion_matrix):

    diagonal_sum = confusion_matrix.trace()

    sum_of_all_elements = confusion_matrix.sum()

    return diagonal_sum / sum_of_all_elements
print(f"Accuracy of model : {accuracy(cm)}")
predictions = classifier.predict(X_test)
output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
output.to_csv('my_submission.csv', index=False)