# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
# Any results you write to the current directory are saved as output.

__author__ = 'Ashish Dutt'
import pandas as pd
import numpy as np
import re, csv

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
data= pd.read_csv("../input/FoodFacts.csv", low_memory=False)
# Creating a copy of the original dataset as sub. All experiments will be done on sub
sub=data

#Determine the number of rows and columns in the dataset
print (data.shape)
# Print the column headers/headings
names=data.columns.values
print (names)
# print the rows with missing data
print ("The count of rows with missing data: \n", sub.isnull().sum())
# Set the rows with missing data as -1 using the fillna()
sub=sub.fillna(-1)
# Show the Frequency distribution
print ("\n Food brands around the world")
foodBrands=data['brands'].value_counts(sort=True,dropna=False)
print (foodBrands)
foodCategory=data['categories_en'].value_counts(sort=True, dropna=False)
print (foodCategory)
manufacPlace=data['manufacturing_places'].value_counts(sort=True, dropna=False)
print (manufacPlace)
foodCountry=data['countries'].value_counts(sort=True, dropna=False)
print (foodCountry)