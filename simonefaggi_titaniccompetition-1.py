# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# Import train and test dataset
train = pd.read_csv("../input/titanic/train.csv")
test = pd.read_csv("../input/titanic/test.csv")

train_shape = train.shape
test_shape = test.shape

print(train_shape)
print(test_shape)
# Visualize 10 first rows
train.head(10)
import matplotlib.pyplot as plt

# Let's see if gender affect survival
# Use pivot_table() to segment our data by sex and calculate the mean of the column Survived
sex_pivot = train.pivot_table(index="Sex",values="Survived")

print(sex_pivot)

# sex_pivot.plot.bar()
# plt.show()
# Let's see if Pclass (ticket class) affect survival
# Use pivot_table() as before

pclass_pivot = train.pivot_table(index="Pclass", values="Survived")
print(pclass_pivot)
# Take a look to the "age" column
train["Age"].describe()
# Age is a continuous numerical column
# Let's use histograms to visualize how age affect survival --> we'll se it's not so good

# Survived histogram
# Get rows for which people survived
survived = train[train["Survived"] == 1]
# Histogram
survived["Age"].plot.hist(alpha=0.5,color='red',bins=50)

# Died histogram
# Get rows for which people died
died = train[train["Survived"] == 0]
# Histogram
died["Age"].plot.hist(alpha=0.5,color='blue',bins=50)

# Plot
plt.legend(['Survived','Died'])
plt.show()
# Let's use categorical features!
# Separate this continuous feature into a categorical feature by dividing it into ranges.
# Use pandas.cut()
# Takes 2 required arguments (dataframe, where_to_cut)
# We use one more argument label_names, this allows to have more readable features

# !!!! IMPORTANT !!!!!!
# Any change we make to the train data, we also need to make to the test data, so let's define a function

def process_age(df, where_to_cut, label_names):
    # Handle missing values
    df["Age"] = df["Age"].fillna(-0.5)
    df["Age_categories"] = pd.cut(df["Age"], where_to_cut, labels=label_names)
    return df
    
# from -1 to 0 --> Missing
# from 0 to 5 --> Infant
# ......
# from 60 to 100 --> Senior
cut_points = [-1, 0, 5, 12, 18, 35, 60, 100]
# Define label names
label_names = ["Missing", 'Infant', "Child", 'Teenager', "Young Adult", 'Adult', 'Senior']

train = process_age(train, cut_points, label_names)
test = process_age(test, cut_points, label_names)

# Visualize data using pivot
age_categories_pivot = train.pivot_table(index="Age_categories", values="Survived")
age_categories_pivot.plot.bar()
plt.plot()
# Create dummies Columns
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

train.head()
from sklearn.linear_model import LogisticRegression

columns = ['Pclass_1', 'Pclass_2', 'Pclass_3', 'Sex_female', 'Sex_male',
       'Age_categories_Missing','Age_categories_Infant',
       'Age_categories_Child', 'Age_categories_Teenager',
       'Age_categories_Young Adult', 'Age_categories_Adult',
       'Age_categories_Senior']

# Create the LogisitcRegression model
lr = LogisticRegression()

# Split train set in X and y
# X = features (age, sex, ...) 
# y = target (survived)

train_X = train[columns]
train_y = train["Survived"]

# Train the model
lr.fit(train_X, train_y)

# Predict on test set
test_pred = lr.predict(test[columns])
print(test_pred)
# Preparing data for submission
test_ids = test["PassengerId"]
submission_df = {"PassengerId": test_ids,
                 "Survived": test_pred}

submission = pd.DataFrame(submission_df)

print(submission)

submission.to_csv('titanic_submission.csv', index=False)
