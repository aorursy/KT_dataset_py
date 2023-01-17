import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import random

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os

print(os.listdir("../input"))
# Reading Dataset



Zoo=pd.read_csv("../input/zoo.csv")

Zoo.head()
# Task Number 1

# Aggregating Data

# 1) Count

# 2) Sum

# 3) Min, Max

# 4) Mean, Median

# 5) Grouping



Zoo.count()
Zoo[['animal']].count()   # Count For Animal Only
Zoo.animal.count()
Zoo.water_need.sum()
Zoo.water_need.max()
Zoo.water_need.min()
Zoo.groupby('animal').mean()    # Run aggregation function for all columns
Zoo.groupby('animal').mean().water_need
Zoo.groupby('animal').mean()[['water_need']]
#'water_need': lambda x: x.max() - x.min()

def max_min(x):

    return x.max() - x.min()





Zoo.groupby('animal').agg({'uniq_id':'count', 

                           'water_need':['mean', 'max', 'min', max_min]

                         })
Zoo[Zoo.animal == 'zebra'].groupby('animal').agg({'uniq_id':'count', 

                           'water_need':['mean', 'max', 'min', max_min]

                         })
# Transpose of a Dataset

# Create the index 

Rows = ['Row_'] * 22

Rows = list(Rows)

Numbers = list(range(1,23))

concat_func = lambda x,y: x + "" + str(y)

index_ = list(map(concat_func,Rows,Numbers))



# Apply Index to Zoo Data

Zoo.index = index_ 

print(Zoo) 

#index_ = map(lambda (x,y): zip(Rows,Numbers))

#print(index_)
Zoo_Tr = Zoo.transpose()

Zoo_Tr
data = {'ID':[1,2,3,4,5,6,7], 'Year':[2016,2016,2016,2016,2017,2017,2017], 'Jan_salary':[4500,4200,4700,4500,4200,4700,5000], 'Feb_salary':[3800,3600,4400,4100,4400,5000,4300], 'Mar_salary':[3700,3800,4200,4700,4600,5200,4900]}

df = pd.DataFrame(data)

df
# Wide to Long Dataset



melted_df = pd.melt(df,id_vars=['ID','Year'],

                        value_vars=['Jan_salary','Feb_salary','Mar_salary'],

                        var_name='month',value_name='salary')

melted_df
# Long To Wide

Casted_df = melted_df.pivot_table(index=['ID','Year'], columns='month', values='salary', aggfunc='first')

Casted_df
# Crosstab

pd.crosstab(index=[melted_df['Year']], columns=[melted_df['month']])
# head

Zoo.head(10)
# Tail

Zoo.tail(4)
# Select/Drop variables & Selecting Observations

df.loc[:, ['ID', 'Year','Jan_salary']]
df.loc[:3, ['ID', 'Year','Jan_salary']]
df.loc[2:5, ['ID', 'Year','Jan_salary']]
df.drop('Jan_salary', axis=1)
df.drop(df.index[2])
# Rndom sample with replacement



#Ind = random.choices(list(range(1,23)), k=12)     # Size = 12

Ind = random.choices(Zoo.index, k=12)     # Size = 12

Sample_1 = Zoo.loc[Ind,]

Sample_1
# Rndom sample with replacement

Ind = random.sample(list(Zoo.index), k=12)

Sample_2 = Zoo.loc[Ind,]

Sample_2