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
## Load text file via read_csv
pop_df = pd.read_csv("../input/banking.csv")
pop_df
## Get top 5 instances
pop_df.head(5)
## Return columns
pop_df.columns
## N-dimensional representation of DataFrame
## What happens to dtypes?
pop_df.values
## Get column dtypes
pop_df.dtypes
## Assign column to new object
## Is this new object a DataFrame?
income_df = pop_df["Income"]
income_df
## Select column directly using the attribute 
pop_df.Income
## Print statement
print("Mean Income: ", income_df.mean())
## Additional print statements
print("Means: \n", pop_df[["Age","Education","Income"]].mean())
print("\n")
print("Standard Deviations: \n", pop_df[["Age","Education","Income"]].std())
## Examine descriptive statistics
pop_df.describe() #include=all for categorical variables
## Can you see anything fishy?
pd.isnull(pop_df.Income)
pop_df[pd.isnull(pop_df.Income)]
pop_df[pd.isnull(pop_df.Wealth)]
pop_df.dropna().describe()
pop_df.describe()
pop_df2 = pop_df.dropna()
pop_df2.describe()
## Finding all instances when the education is above 14
education_above_14 = pop_df2.Education>14
pop_df2[education_above_14].Education.sort_values()
pop_df2[pop_df2.Education>17].Education.sort_values()
pop_df3 = pop_df2.copy()
pop_df3.loc[pop_df3.Education>17] = pop_df2[pop_df2.Education>17]/10
pop_df3.describe()
pop_df3 = pop_df2.copy()
pop_df3.loc[pop_df3.Education>17] = pop_df2.Education.mean()
pop_df3.describe()
pop_df3.corr()
df = pop_df3.copy()
df.Age.hist()
# plot histogram and boxplot for each attribute
for col in df.columns:
    df.hist(col)
    plt.show()
    df.boxplot(col)
    plt.show()
for col in df.columns:
    if col != 'Balance':
        df.plot(x=col, y='Balance', kind="scatter")
        plt.title(col+" and Balance")
        plt.show()