# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
gender_submission_path = "/kaggle/input/titanic/gender_submission.csv"

train_path = "/kaggle/input/titanic/train.csv"

test_path = "/kaggle/input/titanic/test.csv"



gender_submission = pd.read_csv(gender_submission_path)

train_data = pd.read_csv(train_path)

test_data = pd.read_csv(test_path)
gender_submission.head()
train_data.head()
train_data.info()
test_data.info()
train_data.describe()
train_data.describe(include = ['O'])
rate_cabin_Nan = 1- (train_data.Cabin.count()/len(train_data.Cabin))

print("% of NaN values: " ,rate_cabin_Nan)
combine = [train_data, test_data]



# Because Sex has no missing values I can transform it directly

for dataset in combine:

    dataset["Sex"] = dataset["Sex"].map( {"female": 1, "male" : 0} ).astype(int)

    

train_data.head()
# I will change the missing values form Emarked with the most frequent value

frequent_port = train_data.Embarked.dropna().mode()[0]

frequent_port
for dataset in combine:

    dataset["Embarked"] = dataset["Embarked"].fillna(frequent_port)

    

train_data.describe(include = ['O'])
train_data["Embarked"].unique()
# Now I transform it to a numerical feature

for dataset in combine:

    dataset["Embarked"] = dataset["Embarked"].map( {"S" : 0, "C" : 1, "Q" : 2} ).astype(int)

    

train_data.head()
y = train_data.Survived



features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]



X = train_data[features]

X_test = test_data[features]
from sklearn.impute import SimpleImputer



# Imputation

my_imputer = SimpleImputer()

imputed_X = pd.DataFrame(my_imputer.fit_transform(X))

imputed_X_test = pd.DataFrame(my_imputer.transform(X_test))



# Imputation removed column names; put them back

imputed_X.columns = X.columns

imputed_X_test.columns = X_test.columns



imputed_X.describe()
from sklearn.ensemble import RandomForestClassifier



model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)

model.fit(imputed_X, y)

predictions = model.predict(imputed_X_test)



output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})

output.to_csv('my_submission.csv', index=False)

print("Your submission was successfully saved!")