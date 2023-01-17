# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

from sklearn.model_selection import cross_val_score

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
# Loading Training and Test files into Memory

test = pd.read_csv('../input/test.csv')

train = pd.read_csv('../input/train.csv')
# Calculate the number of rows and columns in Train and Test datasets

print("Shape of Train Dataset:" + str(train.shape)),

print("Shape of Test Dataset:" + str(test.shape))
# descriptions contained in that data

"""

PassengerID - A column added by Kaggle to identify each row and make submissions easier

Survived - Whether the passenger survived or not and the value we are predicting (0=No, 1=Yes)

Pclass - The class of the ticket the passenger purchased (1=1st, 2=2nd, 3=3rd)

Sex - The passenger's sex

Age - The passenger's age in years

SibSp - The number of siblings or spouses the passenger had aboard the Titanic

Parch - The number of parents or children the passenger had aboard the Titanic

Ticket - The passenger's ticket number

Fare - The fare the passenger paid

Cabin - The passenger's cabin number

Embarked - The port where the passenger embarked (C=Cherbourg, Q=Queenstown, S=Southampton)

"""
# First 5 rows of the train dataset

train.head(5)
# Calculate the mean of 'Sex ' with target variable 'Survived' to see the effect of Sex on survival

sex_pivot = train.pivot_table(index="Sex",values="Survived")

sex_pivot.plot.bar()

plt.show();
# Calculate the mean of 'Pclass' with target variable 'Survived'

pclass_pivot = train.pivot_table(index='Pclass', values='Survived')

pclass_pivot.plot.bar()

plt.show()
# Age 

train["Age"].describe()
# Age has some missing values

train.shape
# Relationship of people who died and survived in the age group

survived = train[train["Survived"] == 1]

died = train[train["Survived"] == 0]

survived["Age"].plot.hist(alpha=0.5,color='red',bins=50)

died["Age"].plot.hist(alpha=0.5,color='blue',bins=50)

plt.legend(['Survived','Died'])

plt.show()
# Function to process the age into various groups : Missing , Infant , Child , Teenager , Young Adult

# Adult & Senior

def process_age(df,cut_points,label_names):

    df["Age"] = df["Age"].fillna(-0.5)

    df["Age_categories"] = pd.cut(df["Age"],cut_points,labels=label_names)

    return df

cut_points = [-1,0,5,12,18,35,60,100]

label_names = ["Missing","Infant","Child",'Teenager','Young Adult',"Adult",'Senior']

train = process_age(train,cut_points,label_names)

test = process_age(test,cut_points,label_names)

# Pivoting and plotting the Age Categories

train_pivot = train.pivot_table(index='Age_categories', values = 'Survived')

train_pivot.plot.bar()

plt.show()
train["Pclass"].value_counts()
# Creating Dummies for Pclass, Age & Sex

def create_dummies(df,column_name):

    dummies = pd.get_dummies(df[column_name],prefix=column_name)

    df = pd.concat([df,dummies],axis=1)

    return df



train = create_dummies(train,"Pclass")

test = create_dummies(test,"Pclass")

train = create_dummies(train,"Sex")

test = create_dummies(test,"Sex")

train = create_dummies(train,"Age_categories")

test = create_dummies(test,"Age_categories")
train.head(1)
# Training the Model

lr = LogisticRegression()
columns = ['Pclass_1', 'Pclass_2', 'Pclass_3', 'Sex_female', 'Sex_male',

       'Age_categories_Missing','Age_categories_Infant',

       'Age_categories_Child', 'Age_categories_Teenager',

       'Age_categories_Young Adult', 'Age_categories_Adult',

       'Age_categories_Senior']
#  Kaggle 'test' data holdout data

holdout = test
# Train test split on the train dataset

all_X = train[columns]

all_y = train['Survived']

train_X,test_X,train_y,test_y = train_test_split(all_X,all_y, test_size =0.2,random_state =0)
# fit Logistic Regression on the train_test dataset

lr.fit(train_X, train_y)

predictions = lr.predict(test_X)
accuracy = accuracy_score(test_y, predictions)
# overfitting?

accuracy
# check with k-fold cross validation

scores = cross_val_score(lr, all_X,all_y, cv=10)

cross_validation_accuracy = np.mean(scores)

print('K-Fold Cross Validation Scores: \n' + str(scores))

print('Average accuracy: \n' + str(cross_validation_accuracy))

# Using the model to train and test on the Kaggle test dataset(holdout)

lr = LogisticRegression()

lr.fit(all_X,all_y)
# Making prediction

holdout_predictions =lr.predict(holdout[columns])
# saving file as csv

holdout_ids = holdout["PassengerId"]

submission_df = {"PassengerId": holdout_ids,

                 "Survived": holdout_predictions}

submission = pd.DataFrame(submission_df)

submission.to_csv('submission.csv',index=False)