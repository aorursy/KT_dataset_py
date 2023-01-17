# Importing linear algebra

import numpy as np 



# Importing data processing, CSV file I/O (e.g. pd.read_csv)

import pandas as pd



# Importing linear regression

from sklearn.linear_model import LinearRegression





import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# Importing PythonLib for data visualization

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

sns.set()
# Train dataset

train = pd.read_csv("/kaggle/input/titanic/train.csv")

train.head()
# Test dataset

test = pd.read_csv("/kaggle/input/titanic/test.csv")

test.head()
# Calculating percentage of survived women.

women = train.loc[train.Sex == 'female']["Survived"]

rate_women = sum(women)/len(women)



print("% of women who survived:", rate_women)
# Calculating percantage of survived men.

men = train.loc[train.Sex == 'male']["Survived"]

rate_men = sum(men)/len(men)



print("% of men who survived:", rate_men)
train.describe()
train.info()
def bar_chart(feature):

    survived = train[train["Survived"]==1][feature].value_counts()

    dead = train[train["Survived"]==0][feature].value_counts()

    df = pd.DataFrame([survived,dead])

    df.index = ['Survived','Dead']

    df.plot(kind='bar',stacked=True, figsize=(10,5))
bar_chart('Sex')
bar_chart('Pclass')
bar_chart('SibSp')
bar_chart('Parch')
bar_chart('Embarked')
bar_chart('Cabin')
train['Sex'].isnull().value_counts() # There are no NaNs
train["sex_cat"] = train["Sex"].astype("category")

train["sex_cat_code"] = train["sex_cat"].cat.codes

train[["Sex", "sex_cat", "sex_cat_code"]]
test["sex_cat"] = test["Sex"].astype("category")

test["sex_cat_code"] = test["sex_cat"].cat.codes

test[["Sex", "sex_cat", "sex_cat_code"]]
train['Age'].isnull().value_counts() # There are 177 NaNs
mean_age = train["Age"].mean()

train["age_filled"] = train["Age"].fillna(mean_age) 
mean_age = test["Age"].mean()

test["age_filled"] = test["Age"].fillna(mean_age) 
train['age_filled'].isnull().value_counts() # Now, there are no NaNs
train["age_filled_cat"] = train["age_filled"].astype("category")

train["age_filled_cat_code"] = train["age_filled_cat"].cat.codes

train[["age_filled", "age_filled_cat", "age_filled_cat_code"]]
test["age_filled_cat"] = test["age_filled"].astype("category")

test["age_filled_cat_code"] = test["age_filled_cat"].cat.codes

test[["age_filled", "age_filled_cat", "age_filled_cat_code"]]
train['Embarked'].isnull().value_counts() # There are 2 NaNs
train["Embarked"].describe() # 'S' is the most common value
train["embarked_filled"] = train["Embarked"].fillna(value="S")

train["embarked_filled"].describe()
test["embarked_filled"] = test["Embarked"].fillna(value="S")

test["embarked_filled"].describe()
train["embarked_filled_cat"] = train["embarked_filled"].astype("category")

train["embarked_filled_cat_code"] = train["embarked_filled_cat"].cat.codes

train[["embarked_filled", "embarked_filled_cat", "embarked_filled_cat_code"]]
test["embarked_filled_cat"] = test["Embarked"].astype("category")

test["embarked_filled_cat_code"] = test["embarked_filled_cat"].cat.codes

test[["Embarked", "embarked_filled_cat", "embarked_filled_cat_code"]]
train["embarked_filled_cat"].cat.categories = ["Cherbourg", "Queenstown", "Southampton"]

train["embarked_filled_cat"]
test["embarked_filled_cat"].cat.categories = ["Cherbourg", "Queenstown", "Southampton"]

test["embarked_filled_cat"]
train['Fare'].isnull().value_counts() # There are no NaNs
train["fare_cat"] = train["Fare"].astype("category")

train["fare_cat_code"] = train["fare_cat"].cat.codes

train[["Fare", "fare_cat", "fare_cat_code"]]
test["fare_cat"] = test["Fare"].astype("category")

test["fare_cat_code"] = test["fare_cat"].cat.codes

test[["Fare", "fare_cat", "fare_cat_code"]]
train['Cabin'].isnull().value_counts() # There are 204 NaNs
train['cabin_multiple'] = train.Cabin.apply(lambda x: 0 if pd.isna(x) else len(x.split(' ')))
train['cabin_multiple'].value_counts()
train['cabin_multiple'].value_counts()

train["cabin_multiple_cat"] = train["cabin_multiple"].astype("category")

train["cabin_multiple_cat_code"] = train["cabin_multiple_cat"].cat.codes

train[["cabin_multiple", "cabin_multiple_cat", "cabin_multiple_cat_code"]]
test['cabin_multiple'] = test.Cabin.apply(lambda x: 0 if pd.isna(x) else len(x.split(' ')))
test['cabin_multiple'].value_counts()
test['cabin_multiple'].value_counts()

test["cabin_multiple_cat"] = test["cabin_multiple"].astype("category")

test["cabin_multiple_cat_code"] = test["cabin_multiple_cat"].cat.codes

test[["cabin_multiple", "cabin_multiple_cat", "cabin_multiple_cat_code"]]
pd.pivot_table(train, index = 'Survived', columns = 'cabin_multiple_cat', values = 'Ticket' ,aggfunc ='count')
train['SibSp'].isnull().value_counts() # There are no NaNs
train['Parch'].isnull().value_counts() # There are no NaNs
train["relatives"] = train["SibSp"] + train["Parch"]

train[["SibSp", "Parch", "relatives"]].head(10)
test["relatives"] = test["SibSp"] + test["Parch"]

test[["SibSp", "Parch", "relatives"]].head(10)
train['Ticket'].isnull().value_counts() # There are no NaNs
train['ticket_cat']=train['Ticket'].astype('category')

train['ticket_cat_codes']=train['ticket_cat'].cat.codes

train[['Ticket','ticket_cat','ticket_cat_codes']]
test['ticket_cat']=test['Ticket'].astype('category')

test['ticket_cat_codes']=test['ticket_cat'].cat.codes

test[['Ticket','ticket_cat','ticket_cat_codes']]
from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression



y = train["Survived"]



features = ["Sex", "relatives", "fare_cat_code", "Pclass", "embarked_filled"]

features1= ["ticket_cat_codes", "age_filled_cat_code", "Sex"] 



X = pd.get_dummies(train[features])

X1 = pd.get_dummies(train[features1])

X_test = pd.get_dummies(test[features])

X1_test = pd.get_dummies(test[features1])



clf = LogisticRegression(random_state=123)

clf.fit(X1,y)

y_pred=clf.predict(X1_test)

clf.score(X1,y)



model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)

model.fit(X, y)

predictions = model.predict(X_test)



output = pd.DataFrame({'PassengerId': test.PassengerId, 'Survived': predictions})

output.to_csv('my_submission.csv', index=False)

print("Your submission was successfully saved!")
