# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import classification_report

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import AdaBoostClassifier

from sklearn.ensemble import VotingClassifier



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
#Adding the mean Age to the dataframe, where no Age is present

m_age_train = train_data["Age"].mean()

train_data["Age"] = train_data["Age"].fillna(m_age_train)



m_age_test = test_data["Age"].mean()

test_data["Age"] = test_data["Age"].fillna(m_age_test)
#Adding the mean Fare to the dataframe, where no Fare is present

m_fare_train = train_data["Fare"].mean()

train_data["Fare"] = train_data["Fare"].fillna(m_fare_train)



m_fare_test = test_data["Fare"].mean()

test_data["Fare"] = test_data["Fare"].fillna(m_fare_test)
#Adding the frequent Embarked class to the dataframe, where no Embarked is present

e_most_train = train_data["Embarked"].describe()

train_data["Embarked"] = train_data["Embarked"].fillna(e_most_train[2])



e_most_test = test_data["Embarked"].describe()

test_data["Embarked"] = test_data["Embarked"].fillna(e_most_test[2])
#Transform Sex and Embarked to categorical (train and test)

train_data["Sex_cat"] = train_data["Sex"].astype("category")

test_data["Sex_cat"] = test_data["Sex"].astype("category")



train_data["Embarked_cat"] = train_data["Embarked"].astype("category")

test_data["Embarked_cat"] = test_data["Embarked"].astype("category")
#Change the Sex categorical to codes (train and test)

train_data["Sex_codes"] = train_data["Sex_cat"].cat.codes

test_data["Sex_codes"] = test_data["Sex_cat"].cat.codes



#Change the Embarked categorical to codes (train and test)

train_data["Embarked_codes"] = train_data["Embarked_cat"].cat.codes

test_data["Embarked_codes"] = test_data["Embarked_cat"].cat.codes
#Adding new feature Family_size_total from combining SibSp and Parch

train_data["Family_size_total"] = train_data["SibSp"] + train_data["Parch"] + 1

test_data["Family_size_total"] = test_data["SibSp"] + test_data["Parch"] + 1
#Family size to groups sorted by number of total familymembers.

train_data["Family_size_total"] = train_data["Family_size_total"].astype(int)

test_data["Family_size_total"] = test_data["Family_size_total"].astype(int)

def family_range(df):

    df["Family_size_total"].loc[df["Family_size_total"] <= 1 ] = 0

    df["Family_size_total"].loc[(df["Family_size_total"] >= 2) & (df["Family_size_total"] <= 4)] = 1

    df["Family_size_total"].loc[df["Family_size_total"] >= 5] = 2  

family_range(train_data) #run function with train_data

family_range(test_data) #run function with test_data
# Get our y (only for train - Kaggle doesn't give us the test target)

y_train = train_data["Survived"]



# Get our X (train and test)

features = ["Pclass", "Sex_codes", "SibSp", "Parch", "Embarked_codes", "Age", "Fare", "Family_size_total"]

X_test = test_data[features].copy()

X_train = train_data[features].copy()
#Final prediction code#



# Instantiate and fit a classifier

model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1) #randomforrestclassifier

model.fit(X_train, y_train)

model_clf = LogisticRegression(max_iter=500) #logisticregresseionclassifier

model_clf.fit(X_train, y_train)

model_ada = AdaBoostClassifier(n_estimators=500, learning_rate=0.75) #AdaBoostClassifier

model_ada.fit(X_train, y_train)



#combine the classifiers to test them all on the dataset

connect = VotingClassifier(estimators=[('RF', model), ('LR', model_clf), ('AB', model_ada)], 

                          voting='soft',

                          weights=[1, 1, 1])



connect.fit(X_train, y_train)



# Get predictions

predictions = connect.predict(X_test)





# Write to a file (for submission to Kaggle)

output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})

output.to_csv('my_submission.csv', index=False)

print("Your submission was successfully saved!")