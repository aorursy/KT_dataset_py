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
##
##
##
##
##
##
## import pandas_profiling

## pandas_profiling.ProfileReport(titanic_train)
## profile = pandas_profiling.ProfileReport(titanic_train)

## profile.to_file(outputfile="output.html")
##
##
##
##
## df_ordered_by_age = titanic_train.sort_values(by='Age', ascending=False)

##
## title_series = titanic_train['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)

## type(title_series)
##
## columns_X = ['PassengerId', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch',

##                        'Ticket', 'Fare', 'Cabin', 'Embarked', 'Title']

##

##
## type(train_X)
## type(train_Y)
## train_X.tail(7)
## train_Y.tail(7)