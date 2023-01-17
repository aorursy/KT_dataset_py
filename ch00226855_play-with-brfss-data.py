# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib

import matplotlib.pyplot as plt

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data2013 = pd.read_csv("../input/2013.csv")

print("Shape:", data2013.shape)

print("Columns:", data2013.columns)
columns = ['PHYSHLTH', 'SLEPTIM1', 'MENTHLTH', 'SEX']

subdata = data2013[columns]

print(subdata.shape)
# Show the first few rows of the dataset

subdata.head(10)
# Check variable types

subdata.dtypes
# Show the minimum, mean, maximum of the variables

subdata.describe()
# Check if there are missing values

print(subdata.isnull().sum())
# Since there aren't many missing values, we can simply remove those rows

subdata = subdata.dropna()

print("Shape:", subdata.shape)

print(subdata.isnull().sum())
# Show distribution of physical health and mental health

subdata['PHYSHLTH'].hist()
subdata['MENTHLTH'].hist()
# Keep rows with values less than 30

index = (subdata['PHYSHLTH'] <= 30) & (subdata['MENTHLTH'] <= 30)

subdata = subdata[index]

subdata.shape
# Create a scatter plot of physical health and mental health

plt.scatter(subdata['PHYSHLTH'], subdata['MENTHLTH'])
# Use different color for sex

plt.scatter(subdata['PHYSHLTH'], subdata['MENTHLTH'], c=subdata['SEX'])