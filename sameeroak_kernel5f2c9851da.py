# Importing the libraries
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Importing the training dataset
train = pd.read_csv("../input/train.csv")# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

# Viewing the number of rows and columns in the training dataset
train_shape = train.shape
print(train_shape)

test = pd.read_csv("../input/test.csv")

# Viewing the number of rows and columns in the training dataset
test_shape = test.shape
print(test_shape)

train.info()

train.head()

import matplotlib.pyplot as plt
# Calling the pivot_table() function for Sex
sex_pivot = train.pivot_table(index = "Sex", values = "Survived")
sex_pivot.plot.bar()
plt.show()

# Calling the dataframe.pivot_table() function for Pclass
pclass_pivot = train.pivot_table(index = "Pclass", values = "Survived")
pclass_pivot.plot.bar()
plt.show()

# Let's take a look at the Age column using Series.describe()
train["Age"].describe()

# Contains the details of the passengers who survived
survived = train[train["Survived"] == 1]
survived["Age"].plot.hist(alpha=0.5, color="red", bins=50)

# Contains the details fo the passengers who died
died = train[train["Survived"] == 0]
died["Age"].plot.hist(alpha=0.5, color="blue", bins=50)

# Viewing them combined
survived["Age"].plot.hist(alpha=0.5, color="red", bins=50)
died["Age"].plot.hist(alpha=0.5, color="blue", bins=50)
plt.legend(["Survived","Died"])
plt.show()

# Create a function to process the Age column to different categories
def process_age(df, cut_points, label_names):
    # use the pandas.fillna() method to fill all of the missing values with -0.5
    df["Age"] = df["Age"].fillna(-0.5)
    # cuts the Age column using pandas.cut()
    df["Age_categories"] = pd.cut(df["Age"], cut_points, labels=label_names)
    return df

# Cut the Age column into seven segments: Missing, from -1 to 0 Infant, from 0 to 5 Child, from 5 to 12 Teenager, from 12 to 18 
# Young Adult, from 18 to 35 Adult, from 35 to 60 Senior, from 60 to 100
cut_points = [-1, 0, 5, 12, 18, 35, 60, 100]
label_names = ["Missing", "Infant", "Child", "Teenager", "Young Adult", "Adult", "Senior"]

train = process_age(train, cut_points, label_names)
test = process_age(test, cut_points, label_names) 

# Use the pivot_tables() function to plot with Age_categories column
age_categories_pivot = train.pivot_table(index="Age_categories", values = "Survived")
age_categories_pivot.plot.bar()
plt.show()

# value_counts() function is used to get the count of occurence unique values present in the column of the dataset.
train["Pclass"].value_counts()

# pandas.get_dummies() function will generate columns for us.
def create_dummies(df, column_name):
    dummies = pd.get_dummies(df[column_name], prefix=column_name)
    df = pd.concat([df,dummies], axis=1)
    return df

train = create_dummies(train, "Pclass")
test = create_dummies(test, "Pclass")

train.head()

# Similarly for Sex & Age Categories Column
train = create_dummies(train, "Sex")
test = create_dummies(test, "Sex")

train = create_dummies(train, "Age_categories")
test = create_dummies(test, "Age_categories")

train.head()
