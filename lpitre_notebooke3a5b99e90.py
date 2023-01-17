# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

import pandas as pd # learn more: https://python.org/pypi/pandas



#load the csv file

reviews = pd.read_csv("../input/ign.csv")



#print out the first five rows of data with the pandas head() function

print(reviews.head())



#print out the number or rows and columns in our csv file

print("\nshape: ", reviews.shape)
