import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
#Load the training data

train_df = pd.read_csv("../input/train.csv")

test_df = pd.read_csv("../input/test.csv")

#Let's see what it looks like...

print(train_df.head())
#Remove unnecessary features

train_df = train_df.drop(["PassengerId", "Name", "Ticket"], axis=1)

test_df = test_df.drop(["Name", "Ticket"], axis=1)

#Let's convert a few things to be dummy variables..

sex = pd.get_dummies(train_df["Sex"])

#train_df = train_df.drop(["Sex"])

train_df = pd.concat([train_df, sex], axis = 1, join = "inner")
train_df.drop(["Sex"], axis=1)

train_df.head()
#Convert Age to Child/Adult

train_df.loc[:, "Child"] = train_df["Age"]