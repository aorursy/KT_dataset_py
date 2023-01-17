import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# read data
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv") 

# save IDs for submission.csv 
test_ids = test["PassengerId"]
# exploratory data analysis
#train.describe()
#test.describe()
# data processing

# fill in nan values for "Age" and "Cabin" column
train["Age"] = train["Age"].fillna(value=-0.5)
test["Age"] = test["Age"].fillna(value=-0.5)

train['Cabin'] = train['Cabin'].fillna(value='Z')
test['Cabin'] = test['Cabin'].fillna(value='Z')

# pre-partitions the 'Cabin' values
for i in range(len(train)):
    train.loc[i, 'Cabin'] = train.loc[i, 'Cabin'][0]
for j in range(len(test)):
    test.loc[j, 'Cabin'] = test.loc[j, 'Cabin'][0]

def partition(df, column, cut_points, label_names):
    df[column + "_Categories"] = pd.cut(df[column], cut_points, labels=label_names)
    return df

age_cut_points = [-1, 0, 14, 30, 40, 50, 60, 100]
age_label_names = ["Missing", "Children", "Young Adult", "Adult", "Middle Age", "Old Adult", "Senior"]

# partition the "Age" column into categories
train = partition(train, "Age", age_cut_points, age_label_names)
test = partition(test, "Age", age_cut_points, age_label_names)

sib_cut_points = [-1, 0.5, 2, 9]
sib_label_names = ['0 Siblings/Spouse', '1 Sibling/Spouse', '>1 Siblings']

# partition the "SibSp" column into categories
train = partition(train, "SibSp", sib_cut_points, sib_label_names)
test = partition(test, "SibSp", sib_cut_points, sib_label_names)

parch_cut_points = [-1, 0.5, 2, 7]
parch_label_names = ['0 Parent/Child', '1 Parent/Child', '>1 Parent/Children']

# partition the "Parch" column into categories
train = partition(train, "Parch", parch_cut_points, parch_label_names)
test = partition(test, "Parch", parch_cut_points, parch_label_names)
# generate categorical data

def create_dummies(df, column_name):
    dummies = pd.get_dummies(df[column_name], prefix=column_name)
    df = pd.concat([df, dummies], axis=1)
    return df

train = create_dummies(train,"Pclass")
test = create_dummies(test,"Pclass")
train = create_dummies(train,"Sex")
test = create_dummies(test,"Sex")
train = create_dummies(train,"Age_Categories")
test = create_dummies(test,"Age_Categories")
train = create_dummies(train, "Embarked")
test = create_dummies(test, "Embarked")
train = create_dummies(train, "SibSp_Categories")
test = create_dummies(test, "SibSp_Categories")
train = create_dummies(train, "Parch_Categories")
test = create_dummies(test, "Parch_Categories")
train = create_dummies(train, "Cabin")
test = create_dummies(test, "Cabin")
#train.head()
train = train.drop(columns=['PassengerId', "Pclass", 'Name', "Sex", "Age", 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', "Embarked", "Age_Categories", 'SibSp_Categories', 'Parch_Categories', 'Cabin_T'])
test = test.drop(columns=['PassengerId', "Pclass", 'Name', "Sex", "Age", 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', "Embarked", "Age_Categories", 'SibSp_Categories', 'Parch_Categories'])
#test.head()
data = train.loc[:, 'Pclass_1':'Cabin_Z']
target = train.loc[:, "Survived"]
# split data
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    data, target, stratify=target, random_state=100)
# build sample model and evaluate accuracy
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression(C=100).fit(X_train, y_train)

# score
print(logreg.score(X_train, y_train))
print(logreg.score(X_test, y_test))
# build actual model to predict for test.csv

logreg = LogisticRegression().fit(data, target)

prediction = logreg.predict(test)
# submission
submission_df = {"PassengerId": test_ids,
                 "Survived": prediction}
submission = pd.DataFrame(submission_df)
submission.to_csv('titanic_submission_test.csv', index=False)
