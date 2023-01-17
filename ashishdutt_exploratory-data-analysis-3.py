# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

__author__ = 'Ashish Dutt'
import re, csv
import pandas as pd 
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
#data= pd.read_csv("../input/
#sub=data
# Any results you write to the current directory are saved as output.
#print (sub.shape)
data= pd.read_csv("../input/train.csv") # read the data
print (data.shape)
# Print the column headers/headings
names=data.columns.values
print (names)
# print the rows with missing data
print ("The count of rows with missing data: \n", data.isnull().sum())
# make a copy of the data
sub=data
# Set the rows with missing data as -1 using the fillna()
sub=sub.fillna(-1)
sns.countplot(data.AnimalType, palette='Set3')
sns.countplot(data.OutcomeType, palette='Set3')
print (data.Breed)
