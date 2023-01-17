import numpy as np

import pandas as pd

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
import pandas as pd



test = pd.read_csv("/kaggle/input/titanic/test.csv")

test_shape = test.shape

print(test_shape)
train = pd.read_csv("/kaggle/input/titanic/train.csv")

train_shape = train.shape

print(train_shape)
train.head(10)
test.head(10)
women = train.loc[train.Sex == 'female']["Survived"]

rate_women = sum(women)/len(women)



print("% of women who survived:", rate_women)
men = train.loc[train.Sex == 'male']["Survived"]

rate_men = sum(men)/len(men)



print("% of men who survived:", rate_men)
import matplotlib.pyplot as plt



sex_pivot = train.pivot_table(index="Sex",values="Survived")

sex_pivot
pclass_pivot = train.pivot_table(index="Pclass",values="Survived")

pclass_pivot.plot.bar()

plt.show()
train['Age'].describe()
train[train["Survived"] == 1]
survived = train[train["Survived"] == 1]

died = train[train["Survived"] == 0]

survived["Age"].plot.hist(alpha=1,color='orange',bins=50)

died["Age"].plot.hist(alpha=0.6,color='green',bins=50)

plt.legend(['Survived','Died'])

plt.show()
def process_age(df,cut_points,label_names):

    df["Age"] = df["Age"].fillna(-0.5)

    df["Age_categories"] = pd.cut(df["Age"],cut_points,labels=label_names)

    return df



cut_points = [-1,0, 5, 12, 18, 35, 60, 100]

label_names = ["Missing", 'Infant', "Child", 'Teenager', "Young Adult", 'Adult', 'Senior']



train = process_age(train,cut_points,label_names)

test = process_age(test,cut_points,label_names)



age_cat_pivot = train.pivot_table(index="Age_categories",values="Survived")

age_cat_pivot.plot.bar()

plt.show()
train['Pclass'].value_counts()
column_name = "Pclass"

df = train

dummies = pd.get_dummies(df[column_name],prefix=column_name)

dummies.head()
def create_dummies(df,column_name):

    dummies = pd.get_dummies(df[column_name],prefix=column_name)

    df = pd.concat([df,dummies],axis=1)

    return df



train = create_dummies(train,"Pclass")

test = create_dummies(test,"Pclass")

train.head()
train = create_dummies(train,"Sex")

test = create_dummies(test,"Sex")

train = create_dummies(train,"Age_categories")

test = create_dummies(test,"Age_categories")
train.head()
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
columns = ['Pclass_1', 'Pclass_2', 'Pclass_3', 'Sex_female', 'Sex_male',

       'Age_categories_Missing','Age_categories_Infant',

       'Age_categories_Child', 'Age_categories_Teenager',

       'Age_categories_Young Adult', 'Age_categories_Adult',

       'Age_categories_Senior']



from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()

lr.fit(train[columns], train['Survived'])
from sklearn.model_selection import train_test_split



columns = ['Pclass_2', 'Pclass_3', 'Sex_male']



all_X = train[columns]

all_y = train['Survived']



train_X, test_X, train_y, test_y = train_test_split(

    all_X, all_y, test_size=0.2,random_state=0)
holdout = test 

from sklearn.model_selection import train_test_split

columns = ['Pclass_1', 'Pclass_2', 'Pclass_3', 'Sex_female', 'Sex_male',

       'Age_categories_Missing','Age_categories_Infant',

       'Age_categories_Child', 'Age_categories_Teenager',

       'Age_categories_Young Adult', 'Age_categories_Adult',

       'Age_categories_Senior']

all_X = train[columns]

all_y = train['Survived']

train_X, test_X, train_y, test_y = train_test_split(

    all_X, all_y, test_size=0.2,random_state=0)
train_X.shape
lr = LogisticRegression()

lr.fit(train_X, train_y)

predictions = lr.predict(test_X)
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(test_y, predictions)
from sklearn.metrics import accuracy_score
lr = LogisticRegression()

lr.fit(train_X, train_y)

predictions = lr.predict(test_X)

accuracy = accuracy_score(test_y, predictions)

accuracy
from sklearn.metrics import confusion_matrix



conf_matrix = confusion_matrix(test_y, predictions)

pd.DataFrame(conf_matrix, columns=['Survived', 'Died'], index=[['Survived', 'Died']])
from sklearn.model_selection import cross_val_score

import numpy as np



lr = LogisticRegression()

scores = cross_val_score(lr, all_X, all_y, cv=10)

np.mean(scores)
columns = ['Pclass_1', 'Pclass_2', 'Pclass_3', 'Sex_female', 'Sex_male',

       'Age_categories_Missing','Age_categories_Infant',

       'Age_categories_Child', 'Age_categories_Teenager',

       'Age_categories_Young Adult', 'Age_categories_Adult',

       'Age_categories_Senior']

holdout.head()
lr = LogisticRegression()

lr.fit(all_X, all_y)

holdout_predictions = lr.predict(holdout[columns])

holdout_predictions
holdout_ids = holdout["PassengerId"]

submission_df = {"PassengerId": holdout_ids,

                 "Survived": holdout_predictions}

submission = pd.DataFrame(submission_df)
holdout_ids = holdout["PassengerId"]

submission_df = {"PassengerId": holdout_ids,

                 "Survived": holdout_predictions}

submission = pd.DataFrame(submission_df)



submission.to_csv('titanic_submission.csv', index=False)