# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
# Reading data from the CSV 

import csv as csv 

#Import the dataset and do some basic manipulation

traindf = pd.read_csv('../input/train.csv', header=0) 
# We can have a look at the data, shape and types

traindf.dtypes
traindf.info()
traindf.describe
# Get list from pandas DataFrame column headers 

list(traindf)

#type(traindf)
# Get Column data from Pandas Data Frame 

gender =  traindf['Sex']

gender # this is in Series format 

traindf['Survived'].plot.hist()

traindf['Pclass'].value_counts()

gender.value_counts()
# Creating a data set with all female pasenger 

femaledf =  traindf.loc[(traindf.Sex=='female')]

maledf =  traindf.loc[(traindf.Sex=='male')]



# 

traindf['Age'].plot.hist()

maledf['Age'].plot.hist()

femaledf['Age'].plot.hist()
#Pasanger Class distribution - Total, Maele and Female passenger 

traindf['Pclass'].plot.hist()

maledf['Pclass'].plot.hist()

femaledf['Pclass'].plot.hist()
traindf['Survived'].plot.hist()

femaledf['Survived'].plot.hist()

maledf['Survived'].plot.hist()

print("THE END")