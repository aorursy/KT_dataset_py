# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.



# Any results you write to the current directory are saved as output.
# get titanic & test csv files as a DataFrame

titanic = pd.read_csv("../input/train.csv")

test    = pd.read_csv("../input/test.csv")



titanic.head()
titanic.info()

print("----------------------------")

test.info()
titanic_df = titanic_df.drop(['PassengerId','Ticket'], axis=1)
titanic.isnull().sum()