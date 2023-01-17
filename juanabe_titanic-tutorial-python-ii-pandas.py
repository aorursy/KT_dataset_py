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
# load the data

# use Pandas own functions to read or write .csv files

# Pandas infers numerical types (so now not every column is

# a string, have int and floats - df.dtypes )



df = pd.read_csv('../input/train.csv', header=0)



# display: df, def.head(n), df.tail(n)

# use df.info() - tells # rows, # non-null entries per column

# and datatype

# use df.describe() - calculates mean, std, min, max of 

# all numerical columns (left nulls out of calculation)
# df['Age'][0:10] gives same as df.Age[0:10]

# results is a data Series

# df['Age'].mean() gives mean age

# subset of data using: df[ ['Sex', 'Pclass', 'Age'] ]

# filter data: df[df['Age'] > 60] gives passengers over 60

# combine: df[df['Age'] > 60][['Sex', 'Pclass', 'Age', 'Survived']]

# filter missing values: df[df['Age'].isnull()][['Sex', 'Pclass', 'Age', 'Survived']]



# combine criteria:

for i in range(1,4):

    print (i, len(df[ (df['Sex'] == "male") & (df['Pclass'] == i)]) )
# draw a histogram (shortcut to features of matplotlib/pylab packages)

import pylab as P

df['Age'].hist()

P.show()
df['Age'].dropna().hist(bins=16, range=(0,80), alpha=.5)

P.show()
# difficult to run analysis of strings of "male" and "female"

# transform it



# add a column with 4 for every value

df['Gender'] = 4



# use values in 'Sex' coloumn to update 'Gender' column

df['Gender'] = df['Sex'].map( lambda x: x[0].upper() )

# Gender column now has 'M' and 'F'



# but ideally would like Gender as a binary integer

# female = 0; male = 1

df['Gender'] = df['Sex'].map( {'female': 0, 'male': 1} ).astype(int)



# try same for embarked values

df['Embarked'].unique() 
# add new column

df['Origin'] = 4



# 'S' = 0, 'C' = 1, 'Q' = 2, nan=4

df['Origin'] = df['Embarked'].dropna().map( {'S':0, 'C':1, 'Q':2} ).astype(int)



df.head(15)
# deal with missing values of Age

# fill in missing values with guesses

# age histogram positively skewed, so median seems better than mean

# use age typical in each passanger class



median_ages = np.zeros((2,3))



for i in range(0,2):

    for j in range(0,3):

        median_ages[i,j] = df[(df['Gender'] == i) &\

                             (df['Pclass'] == j+1)]['Age'].dropna().median()

        

median_ages
# make a new column where any null ages will be replaced by median for that class

# make a copy of Age

df['AgeFill'] = df['Age']



df[ df['Age'].isnull() ][['Gender', 'Pclass', 'Age', 'AgeFill']].head(10)
# assign median ages

for i in range(0,2):

    for j in range(0,3):

        df.loc[ (df.Age.isnull()) & (df.Gender == i) & (df.Pclass == j+1),\

              'AgeFill'] = median_ages[i,j]

        

      

df[ df['Age'].isnull() ][['Gender','Pclass','Age','AgeFill']].head(10)
# create feature which records whether the Age was originally missing

df['AgeIsNull'] = pd.isnull(df.Age).astype(int)

# Parch = number of parents or children on board

# SibSp = number of siblings or spouses

df['FamilySize'] = df['SibSp'] + df['Parch']



# create new feature

df['Age*Class'] = df.AgeFill * df.Pclass



df['FamilySize'].dropna().hist()

df['Age*Class'].dropna().hist()
# ML techniques don't usually work on strings

# pythin requires data to be an array

# sklearn package not written to use a pandas dataframe

# (1) determine what columns are left which are not numeric

# (2) send the pandas.DataFrame back to a numpy.array



# in pandas, see column types using .info() or df.dtypes

# use .dtypes to show which columns are an object:

df.dtypes[df.dtypes.map(lambda x: x=='object')]
# drop column we will not use:

df = df.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1)



# can also drop Age, as now have AgeFill

df = df.drop(['Age'], axis=1)



# alternative command to srop any rows which still have missing values:

df = df.dropna()
# convert to numpy array

train_data = df.values

train_data
data