# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")

print(train.describe())

print(test.describe())
# Create a copy of test: test_one

test_one = test.copy()



# Initialize a Survived column to 0

test_one['Survived'] = 0



# Set Survived to 1 if Sex equals "female" and print the `Survived` column from `test_one`

test_one['Survived'][test_one['Sex'] == "female"] = 1



test_one = pd.DataFrame(test_one, columns=["PassengerId", "Survived"])



print(test_one.head())

print(test_one.describe())
test_one.to_csv("test_one.csv", index=False)