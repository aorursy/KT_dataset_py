# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



#from subprocess import check_output

#print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.



# Read the files

test = pd.read_csv("../input/test.csv")

train = pd.read_csv("../input/train.csv")



print("Dimensions of train: {}".format(train.shape))

print("Dimensions of test: {}".format(test.shape))
train.head()
# exploring Sex and Pclass by visualizing the data.

import matplotlib.pyplot as plt

%matplotlib inline

sex_pivot = train.pivot_table(index="Sex",values="Survived")

sex_pivot.plot.bar()

plt.show()

# Observe female survival probability is higher than male
# Visualize Pclass column

class_pivot = train.pivot_table(index="Pclass",values="Survived")

class_pivot.plot.bar()

plt.show()

# Observe that elite class has better survival chance
# Look at Age column, notice fractional age

# Also notice, count < 891, indicates missing values.

train["Age"].describe()

# train.isnull().any() # saurabh
# plot of dead and survived

figure = plt.figure(figsize=(15,8)) # saurabh

survived = train[train["Survived"] == 1]

died = train[train["Survived"] == 0]

survived["Age"].plot.hist(alpha=0.5,color='red',bins=50)

died["Age"].plot.hist(alpha=0.5,color='blue',bins=50)

plt.legend(['Survived','Died'])

plt.show()

# Notice in some age ranges more passengers survived 

# where the red bars are higher than the blue bars.
#figure = plt.figure(figsize=(15,8)) # not working, saurabh



# Feature Engineering ('Age' feature) :

# Fill all of the missing Age values with -0.5

# Cuts the Age column into six segments:

# Missing, from -1 to 0

# Infant, from 0 to 5

# Child, from 5 to 12

# Teenager, from 12 to 18

# Young Adult, from 18 to 35

# Adult, from 35 to 60

# Senior, from 60 to 100



# Firstly, any change we make to the train data, 

# we also need to make to the test data, otherwise 

# we will be unable to use our model to make predictions

# for submission.



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
# values in the Pclass columnare 1, 2, and 3

train["Pclass"].value_counts()
# Feature Engineering ("Pclass" feature)

# Remove the "numeric" relationship between the 

# classes, since class "2" isn't double that of

# class "1". Using get_dummies generate new feautures

# Pclass_1, Pclass_2 and Pclass_3

# Sex_male, Sex_female

# Age_categories_Missing, Age_categories_Infant, Age_categories_Child, 

# Age_categories_Teenager, Age_categories_Young Adult, Age_categories_Adult,

# Age_categories_Senior



def create_dummies(df,column_name):

    dummies = pd.get_dummies(df[column_name],prefix=column_name)

    df = pd.concat([df,dummies],axis=1)

    return df



for column in ["Pclass","Sex","Age_categories"]:

    train = create_dummies(train,column)

    test = create_dummies(test,column)



#train.head()

#test.head()
# Train logistic regression model on All the 

# extracted features 



from sklearn.linear_model import LogisticRegression



columns = ['Pclass_1', 'Pclass_2', 'Pclass_3', 'Sex_female', 'Sex_male',

       'Age_categories_Missing','Age_categories_Infant',

       'Age_categories_Child', 'Age_categories_Teenager',

       'Age_categories_Young Adult', 'Age_categories_Adult',

       'Age_categories_Senior']



lr = LogisticRegression()

lr.fit(train[columns], train["Survived"])



# the above trained model has no data to test/evaluate before 

# the submission, So split the train data for training and

# testing.
# Rename the test data set as hold out data 

holdout = test # from now on we will refer to this

               # dataframe as the holdout data



from sklearn.model_selection import train_test_split



all_X = train[columns]

all_y = train['Survived']



# Split train data into train_X, train_y and test_X, test_y

# An 80/20 split

train_X, test_X, train_y, test_y = train_test_split(

    all_X, all_y, test_size=0.20,random_state=0)
# Train the model on the "split" train data (80% partition)

# Evaluate the model on "split" test data (20% partition)

from sklearn.metrics import accuracy_score



lr = LogisticRegression()

lr.fit(train_X, train_y)

predictions = lr.predict(test_X)

accuracy = accuracy_score(test_y, predictions)



print(accuracy)
# Use cross validation for more accurate score

# here 10 fold cv is done

from sklearn.model_selection import cross_val_score



lr = LogisticRegression()

scores = cross_val_score(lr, all_X, all_y, cv=10)

scores.sort()

accuracy = scores.mean()



print(scores)

print(accuracy)
# cv is giving low score so train on all train data

# and discard the cv model. Then use the test data set

# (renamed to "holdout") to create the submissikon file

lr = LogisticRegression()

lr.fit(all_X,all_y)

holdout_predictions = lr.predict(holdout[columns])
# Create the submission file

holdout_ids = holdout["PassengerId"]

submission_df = {"PassengerId": holdout_ids,

                 "Survived": holdout_predictions}

submission = pd.DataFrame(submission_df)
submission.to_csv("gender_submission.csv",index=False)

from subprocess import check_output

print(check_output(["ls", "."]).decode("utf8"))