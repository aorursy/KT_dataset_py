# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# Any results you write to the current directory are saved as output.



# Read the input file and put the data to pandas' dataframe format.

df = pd.read_csv("../input/Iris.csv")
df.head()
df.ix[:,-1].unique()
df.drop('Id', axis=1, inplace=True)
# TODO: Set up for multiple classifier!

# Hint: use 0, 1, 2 to represent three different types of iris.
# TODO: Prepare training/testing data
# TODO: create a classifier

# TODO: make predictions
# TODO: Choose your own performance measures and calculate them
# TODO: Choose appropriate tests