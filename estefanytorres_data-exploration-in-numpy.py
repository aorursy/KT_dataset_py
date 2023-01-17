# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import math
import matplotlib.pyplot as plt
%matplotlib inline

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
## Get working directory
os.getcwd()

## Change working directory
# os.chdir('../input')
## Read file. 
datafile = open("../input/banking.csv")
## First line of the file contains labels for column names
labels_line = datafile.readline().strip()
banking = []
total = 0
for line in datafile.readlines():
    line = line.strip()
    row = line.split(",")
    banking.append(row)
    total += 1
# print(total)
print(labels_line)
print("Total:",total)
## print
banking
# Printing the top5 elements (rows) (but not including) index 5
banking[0:5]
## Convert to numpy array 
bank = np.array(banking, float)
bank
# Here we'll use standard Python list comprehensions
bank = np.array([row for row in banking if not '' in row], float)
bank
## load text can be used if data is clean already
# bank = np.loadtxt('../input/banking.csv', skiprows=1)
# bank
## Examine shape
bank.shape
bank.T.shape
labels_line
Age,Education,Income,HomeVal,Wealth,Balance = bank.T
bank_columns = [Age,Education,Income,HomeVal,Wealth,Balance]
Age
## Print means
print("Mean Age: ", Age.mean())
print("Std: ", Age.std())
## Correlation coefficient
corr_matrix = np.corrcoef(bank.T)
print(corr_matrix)
## Examine descriptive statistics
plt.hist(Age, bins=8, alpha=0.5)
plt.xlabel('Age')
plt.ylabel('Count')
plt.title('Histogram of Age')
plt.grid(True)
# plot histogram and boxplot for each attribute
for col in bank_columns:
    plt.hist(col, bins=8, alpha=0.5)
    plt.grid(True)
    plt.show()