import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_absolute_error

from sklearn.ensemble import RandomForestClassifier



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# For displaying all the rows of the data frame

pd.set_option('display.max_rows', None) # Thanks to @ https://dev.to/chanduthedev/how-to-display-all-rows-from-data-frame-using-pandas-dha



# get training data

train_data = pd.read_csv("/kaggle/input/titanic/train.csv")

train_data.head()
# get test data

test_data = pd.read_csv("/kaggle/input/titanic/test.csv")

test_data.head()
# Describe training data

train_data.describe()
# Describe test data

test_data.describe()
# Check correlation in training data

train_data.corr(method='pearson')
# Look null values for train data

train_data.isnull().sum()
# Look null values for test data

test_data.isnull().sum()
# There is 1 null value for 'Fare' column in test data

# https://dzone.com/articles/pandas-find-rows-where-columnfield-is-null

test_data[test_data["Fare"].isnull()]
# There are 2 null values for 'Embarked' column in train data

# https://dzone.com/articles/pandas-find-rows-where-columnfield-is-null

train_data[train_data["Embarked"].isnull()]
# Description of the people from Southampton who are traveling in the third class(training dataset)

South_3C_train_data = train_data[train_data["Embarked"] == "S"][train_data["Pclass"] == 3]

South_3C_train_data.describe()
# Description of the people from Southampton who are traveling in the third class(test dataset)

South_3C_test_data = test_data[test_data["Embarked"] == "S"][test_data["Pclass"] == 3]

South_3C_test_data.describe()
# Look if there is another person who has the same ticket number with Mr Storey 

train_data[train_data["Ticket"] == "3701"]
# Look if there is another person who has the same ticket number with Mr Storey 

test_data[test_data["Ticket"] == "3701"]
# Who paid £ 7.854200 in Southampton(training dataset)

train_data[train_data["Embarked"] == 'S'][train_data["Fare"] == 7.854200]
# Who paid £ 7.854200 in Southampton(test dataset)

test_data[test_data["Embarked"] == 'S'][test_data["Fare"] == 7.854200]
# Who else has 4 digit tickets, embarked from Southampton and traveling in the third class? 

train_data[[len(x) == 4 for x in train_data["Ticket"]]][train_data["Embarked"] == "S"][train_data["Pclass"] == 3]
# Who else has 4 digit tickets, embarked from Southampton and traveling in the third class? 

test_data[[len(x) == 4 for x in test_data["Ticket"]]][test_data["Embarked"] == "S"][test_data["Pclass"] == 3]
# Are there any tickets that contains the string 3701? May be Some digits in Mr Storey' ticket is missing.

train_data[['3701' in x for x in train_data['Ticket']]]
# Are there any tickets that contains the string 3701? May be Some digits in Mr Storey' ticket is missing.

test_data[['3701' in x for x in test_data['Ticket']]]
# Insert £ 20.2125 for Mr Storey's fare

test_data.loc[152,'Fare'] = 20.2125
# Check Mr Storey

test_data.loc[152]
# Check again

test_data.isnull().sum()
# There are 2 null values for 'Embarked' column in train data

# https://dzone.com/articles/pandas-find-rows-where-columnfield-is-null

train_data[train_data["Embarked"].isnull()]
# Who paid £ 80 in training data 

train_data[train_data["Fare"] == 80]
# Who paid £ 80 in test data

test_data[test_data["Fare"] == 80]
# People who paid more than £ 70 in training data

train_data[train_data["Fare"] > 70]
# People who paid more than £ 70 in test data

test_data[test_data["Fare"] > 70]
# people in Cabins starting with 'B' in training data

train_data[['B' in str(x) for x in train_data['Cabin']]]
# people in Cabins starting with 'B' in test data

test_data[['B' in str(x) for x in test_data['Cabin']]]
# people in Cabins starting with 'B2' in training data

train_data[['B2' in str(x) for x in train_data['Cabin']]]
test_data[['B2' in str(x) for x in test_data['Cabin']]]
# Miss Icard

train_data.loc[61, 'Embarked'] = 'S'

# Mrs Stone

train_data.loc[829, 'Embarked'] = 'S'
# Check Miss Icard

train_data.loc[61]
# Check Mrs Stone

train_data.loc[829]
# Check again

train_data.isnull().sum()
from sklearn.ensemble import RandomForestClassifier



y = train_data["Survived"]



features = ["Pclass", "Sex", "SibSp", "Parch", "Fare", "Embarked"]

X = pd.get_dummies(train_data[features])

X_test = pd.get_dummies(test_data[features])



model = RandomForestClassifier(n_estimators=100, max_depth=7, random_state=1)

model.fit(X, y)

predictions = model.predict(X_test)



output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})

output.to_csv('my_submission.csv', index=False)

print("Your submission was successfully saved!")