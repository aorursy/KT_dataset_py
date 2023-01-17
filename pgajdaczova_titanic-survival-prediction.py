# Load libraries
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
%matplotlib inline

# Load data
test = pd.read_csv("../input/test.csv") # contains data for testing the model
train = pd.read_csv("../input/train.csv") # contains data for defining and fitting the model

# Check out the dimensions of the dataframe
print("Dimensions of train: {}".format(train.shape))
print("Dimensions of test: {}".format(test.shape))
# Let's look at the first rows of the train data set
train.head()
# The column "Survived" contains 0 if passenger did not survive and 1 if they did, we can segment our data by sex and calculate the mean of this column.
# DataFrame.pivot_table()

sex_pivot = train.pivot_table(index="Sex", values="Survived")
sex_pivot.plot.bar()
plt.show()
# Let's classify and pivot the Pclass column as well

class_pivot=train.pivot_table(index="Pclass", values="Survived")
class_pivot.plot.bar()
plt.show()
# Exploring and converting the age column
# The Sex and PClass are CATEGORICAL(meaning that the values represented a few separate options - female/male)

train["Age"].describe()

# there is only 714 values (compared to 814 in the data set) - there must be some missing values
# also this column is a continuous NUMERICAL column
# for continuous numerical values it is good to use histogram
# let's compare those who survived to those who died across different age ranges
survived = train[train["Survived"]==1]
died = train[train["Survived"]==0]

survived["Age"].plot.hist(alpha=0.5, color='red', bins=50)
died["Age"].plot.hist(alpha=0.5, color='blue', bins=50)
plt.legend(['Survived', 'Died'])
plt.show()
# We can try to separate the continuous feature into categorical by splitting it into ranges using the pandas.cut() function
# pandas.cut() has two parameters
    # 1. the column we would like to cut
    # 2. list of numbers which define the boundaries of our cuts

# Next we need to remember to make any changes we perform on the train data also on the test data because otherwise we will not be able to use the model
# And we need to remember to handle the missing values

# We want to create a function that will do two things:
    # 1. Uses the pandas.fillna() method to fill all of the missing values with -0.5
    # 2. Cuts the Age column into six segments

def process_age (df, cut_points, label_names):
    df["Age"] = df["Age"].fillna(-0.5)
    df["Age_categories"] = pd.cut(df["Age"], cut_points, labels=label_names)
    return df

cut_points = [-1, 0, 5, 12, 18, 35, 60, 100]
label_names = ["Missing", "Infant", "Child", "Teenager", "Young Adult", "Adult", "Senior"]

train = process_age(train, cut_points, label_names)
test = process_age(test, cut_points, label_names)

pivot = train.pivot_table(index="Age_categories", values="Survived")
pivot.plot.bar()
plt.show()
# Preparing data for machine learning

# most machine learning models can't understand text labels so we have to convert our values into numbers

# we need to be sure that we don't imply any numeric relationship where there is not one.

# Values in the Pclass column are 1, 2, 3

train["Pclass"].value_counts()
# Now we should remove the relationships between the different types of classes
# Class 2 is not "worth" double what class 1 is and class 3 is not "worth" triple what class 1 is

# To remove this relationship, we can split this column into 3 new columns with 0,1 values
# To do this a function pandas.get_dummies() will help us

def create_dummies(df,column_name):
    dummies = pd.get_dummies(df[column_name], prefix=column_name)  # prefix parameter means what the new dummy columns will be named like
    df = pd.concat([df,dummies], axis=1)  # add the new columns to my dataframe
    return df

for column in ["Pclass", "Sex", "Age_categories"]:
    train = create_dummies(train,column)  # passing into the function parameter the name of the column defined in the for loop each one by one
    test = create_dummies(test,column) # don't forget to apply also to the test data set
# The first model we will use is called Logistic Regression
# This is usually the first model we will train when performing classification

# 1. import the appropriate class from scikit-learn library
from sklearn.linear_model import LogisticRegression

# 2. create object
lr = LogisticRegression()

# 3. fit our model
columns = ["Pclass_2", "Pclass_3", "Sex_male"]
lr.fit(train[columns], train["Survived"])  # x = train[columns] a two-dimensional array (dataframe), y = train["Survived"] a one-dimensional array (series) that we want to predict
# Let's try to train our model with all dummy columns we created

columns = ['Pclass_1', 'Pclass_2', 'Pclass_3', 'Sex_female', 'Sex_male',
       'Age_categories_Missing','Age_categories_Infant',
       'Age_categories_Child', 'Age_categories_Teenager',
       'Age_categories_Young Adult', 'Age_categories_Adult',
       'Age_categories_Senior']

lr = LogisticRegression()
lr.fit(train[columns], train["Survived"])
# Even though we have a test dataframe to test out our model for predictions, that dataframe does not have the column "Survived" against which we could compare how accurate our model is
# Therefore we still need to split our train data into two dataframes to play around with optimization of our model

# We are splitting our train data set into two:
    # 1. one part to train our model (usually 80% of the observations)
    # 2. one part to make predictions (usually 20% of the observations)
    
# Now we need to rename the Kaggle test data as "holdout" data
holdout = test

# Now using scikit-learn library and function train_test_split we are going to split our training data frame
from sklearn.model_selection import train_test_split

all_x = train[columns]
all_y = train["Survived"]

train_x, test_x, train_y, test_y = train_test_split(all_x, all_y, test_size=0.20, random_state=0)
# train_test_split take several parameters:
    # x and y = all the data we want to split (features and target)
    # test_size = we can specify the proportion we want to split the data
    # random_state = let's us split the data everytime the same so we can get replicable results
# train_test_split then returns four objects: train_x, test_x, train_y, test_y
# In the next step we are going to make some predictions and measure their accuracy

# 1. Fit our model again to the split train data
lr = LogisticRegression()

lr.fit(train_x, train_y)

# 2. Make predictions
predictions = lr.predict(test_x)

# 3. Measure our accuracy
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(test_y, predictions)

print(accuracy)
# Since our data set is pretty small, it is very likely that our model is overfitting
# To understand better the performance of our model we can use CROSS VALIDATION
# It means to train and test our model on different SPLITS of our data and then average the accuracy scores

# K-FOLD cross validation
from sklearn.model_selection import cross_val_score

lr = LogisticRegression()
scores = cross_val_score(lr, all_x, all_y, cv=10)
scores.sort()
accuracy = scores.mean()

print(scores)
print(accuracy)
# Let's make our final predictions

lr = LogisticRegression()
lr.fit(all_x, all_y)
holdout_predictions = lr.predict(holdout[columns])

print(holdout_predictions)
# Prepare submission
holdout_ids = holdout["PassengerId"]
submission_df = {"PassengerId": holdout_ids, "Survived": holdout_predictions}
submission = pd.DataFrame(submission_df)

submission.to_csv("submission_df", index=False)