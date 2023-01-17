import pandas as pd

import matplotlib.pyplot as plt

import numpy as np



test = pd.read_csv("../input/test.csv")

train = pd.read_csv("../input/train.csv")
test.head()
train.head()
sex_pivot = train.pivot_table(index="Sex",values="Survived")

sex_pivot.plot.bar()
pclass_pivot = train.pivot_table(index="Pclass",values="Survived")

pclass_pivot.plot.bar()
# looking at age column

train["Age"].describe()
# filtering by Survived column

survived = train[train["Survived"] == 1]

died = train[train["Survived"] == 0]
survived["Age"].plot.hist(alpha=0.5, color="red", bins=50)

died["Age"].plot.hist(alpha=0.5, color="blue", bins=50)

plt.legend(['Survived','Died'])
def process_age(df,cut_points,label_names):

    df["Age"] = df["Age"].fillna(-0.5)

    df["Age_categories"] = pd.cut(df["Age"],cut_points,labels=label_names)

    return df
cut_points = [-1,0,5,12,18,35,60,100]

label_names = ["Missing","Infant","Child","Teenager", "Young Adult", "Adult", "Senior"]



train = process_age(train,cut_points,label_names)

test = process_age(test,cut_points,label_names)
age_categories_pivot = train.pivot_table(index="Age_categories", values="Survived")

age_categories_pivot.plot.bar()
def create_dummies(df, column_name):

    dummies = pd.get_dummies(df[column_name], prefix=column_name)

    df = pd.concat([df,dummies],axis=1)

    return df
train = create_dummies(train,"Pclass")

test = create_dummies(test,"Pclass")
train = create_dummies(train,"Sex")

test = create_dummies(test,"Sex")
train = create_dummies(train,"Age_categories")

test = create_dummies(test,"Age_categories")
train.head()
test.head()
# training our first model using LogisticRegression

from sklearn.linear_model import LogisticRegression



columns = ['Pclass_1', 'Pclass_2', 'Pclass_3', 'Sex_female', 'Sex_male',

       'Age_categories_Missing','Age_categories_Infant',

       'Age_categories_Child', 'Age_categories_Teenager',

       'Age_categories_Young Adult', 'Age_categories_Adult',

       'Age_categories_Senior']



lr = LogisticRegression()

lr.fit(train[columns], train['Survived'])
# spliting training data in train and test

holdout = test # test data from kaggle now will be called holdout

from sklearn.model_selection import train_test_split

all_x = train[columns]

all_y = train["Survived"]



train_x, test_x, train_y, test_y = train_test_split(all_x, all_y, test_size=0.2,random_state=0)
from sklearn.metrics import accuracy_score



lr = LogisticRegression()

lr.fit(train_x, train_y)

predictions = lr.predict(test_x)

accuracy = accuracy_score(test_y, predictions)

print(accuracy)

# cross validation using k-fold

from sklearn.model_selection import cross_val_score



lr = LogisticRegression()

scores = cross_val_score(lr, all_x, all_y, cv=10)

accuracy = np.mean(scores)



print(scores)

print(accuracy)
lr = LogisticRegression()

lr.fit(all_x[columns], all_y)

holdout_predictions = lr.predict(holdout[columns])

print(holdout_predictions)
# creation submission data

holdout_ids = holdout["PassengerId"]

submission_df = {"PassengerId": holdout_ids,

                 "Survived": holdout_predictions}

submission = pd.DataFrame(submission_df)
submission.to_csv("submission.csv", index=False)