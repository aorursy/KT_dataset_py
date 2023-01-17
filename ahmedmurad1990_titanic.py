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
import pandas as pd

test = pd.read_csv("../input/titanic/test.csv")

train = pd.read_csv("../input/titanic/train.csv")

print("Dimensions of train: {}".format(train.shape))

print("Dimensions of test: {}".format(test.shape))
train.head()
import matplotlib.pyplot as plt



sex_pivot = train.pivot_table(index="Sex",values="Survived")

sex_pivot.plot.bar()

plt.show()
class_pivot = train.pivot_table(index="Pclass",values="Survived")

class_pivot.plot.bar()

plt.show()
train["Age"].describe()

survived = train[train["Survived"] == 1]

died = train[train["Survived"] == 0]

survived["Age"].plot.hist(alpha=0.5,color='red',bins=50)

died["Age"].plot.hist(alpha=0.5,color='blue',bins=50)

plt.legend(['Survived','Died'])

plt.show()
def process_age(df,cut_points,label_names):

    df["Age"] = df["Age"].fillna(-0.5)

    df["Age_categories"] = pd.cut(df["Age"],cut_points,labels=label_names)

    return df



cut_points = [-1,0,5,12,18,35,60,100]

label_names = ["Missing","Infant","Child","Teenager","Young Adult","Adult","Senior"]

train = process_age(train,cut_points,label_names)

test = process_age(test,cut_points,label_names)



pivot = train.pivot_table(index="Age_categories",values='Survived')

pivot.plot.bar()

plt.show()
train["Pclass"].value_counts()

def create_dummies(df,column_name):

    dummies = pd.get_dummies(df[column_name],prefix=column_name)

    df = pd.concat([df,dummies],axis=1)

    return df



for column in ["Pclass","Sex","Age_categories"]:

    train = create_dummies(train,column)

    test = create_dummies(test,column)
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()

columns = ['Pclass_2', 'Pclass_3', 'Sex_male']

lr.fit(train[columns], train['Survived'])
from sklearn.linear_model import LogisticRegression



columns = ['Pclass_1', 'Pclass_2', 'Pclass_3', 'Sex_female', 'Sex_male',

       'Age_categories_Missing','Age_categories_Infant',

       'Age_categories_Child', 'Age_categories_Teenager',

       'Age_categories_Young Adult', 'Age_categories_Adult',

      'Age_categories_Senior']



lr = LogisticRegression()

lr.fit(train[columns], train["Survived"])
LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,

          intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,

          penalty='l2', random_state=None, solver='liblinear', tol=0.0001,

          verbose=0, warm_start=False)
holdout = test # from now on we will refer to this

               # dataframe as the holdout data



from sklearn.model_selection import train_test_split



all_X = train[columns]

all_y = train['Survived']



train_X, test_X, train_y, test_y = train_test_split(

    all_X, all_y, test_size=0.20,random_state=0)
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

print(accuracy)
from sklearn.model_selection import cross_val_score



lr = LogisticRegression()

scores = cross_val_score(lr, all_X, all_y, cv=10)

scores.sort()

accuracy = scores.mean()



print(scores)

print(accuracy)
lr = LogisticRegression()

lr.fit(all_X,all_y)

holdout_predictions = lr.predict(holdout[columns])
holdout_ids = holdout["PassengerId"]

submission_df = {"PassengerId": holdout_ids,

                 "Survived": holdout_predictions}

submission = pd.DataFrame(submission_df)
submission.to_csv("submission.csv",index=False)
