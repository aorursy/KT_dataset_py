# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# Combine two python lists

list_a = [1, 2, 3, 4]

list_b = ['a', 'b', 'c', 'd']

print('list_a + list_b:', list_a + list_b)



# a list can by multiplied by a number

print('list_a * 4:', list_a * 4)



# print a line made of '-'

print('-' * 40)
# Combine numpy arrays

vec1 = np.array([1, 3, 5, 7, 9])

vec2 = np.array([11, 33, 55, 77, 99])



# Use np.hstack() to concatenate arrays horizontally

print('np.hstack([vec1, vec2]):\n',

      np.hstack([vec1, vec2]))



# Use np.vstack() to concatenate arrays vertically

print('np.vstack([vec1, vec2]):\n',

      np.vstack([vec1, vec2]))



# Use np.c_ and np.r_ to do combination

print('np.c_[vec1, vec2]:')

print(np.c_[vec1, vec2])



print(np.c_[vec1, vec2].T)



print('np.r_[vec1, vec2]:')

print(np.r_[vec1, vec2])



# Use np.concatenate([vec1, vec2])

print('np.concatenate([vec1, vec2]):')

print(np.concatenate([vec1, vec2]))
# Combining Pandas DataFrame:

df1 = pd.DataFrame({'Name': ['Alice', 'Bob', 'Clare'],

                    'Attendance': [9, 8, 7]})

df1
df2 = pd.DataFrame({'Name': ['Alice', 'Bob', 'Clare'],

                    'Final Score': [95, 85, 90]})

df2
# Three methods to combine data frames:

# pd.merge()

# pd.DataFrame.join()

# pd.concat()



pd.concat([df1, df2])
df1.join(df2, lsuffix='_df1', rsuffix='_df2')
df3 = pd.DataFrame(df2.values,

                   columns=['Name', 'Score'],

                   index=['Alice', 'Bob', 'Clare'])

df3 = df3.drop('Name', axis=1)

df3
df1.join(df3, on='Name')
# The how parameter can take value 'inner', 'outer', 'left', or 'right'

df4 = pd.DataFrame({'Name': ['Alice', 'Bob', 'Clare', 'Doug'],

                    'Attendance': [9, 8, 7, 6]})

df4
df5 = pd.DataFrame({'Name': ['Alice', 'Bob', 'Clare', 'Ed'],

                    'Final Score': [95, 85, 90, 70]},

                   index=[0, 1, 2, 4])

df5
df4.join(df5, lsuffix='_df4')
df4.join(df5, lsuffix='_df4', how='outer')
df4.join(df5, lsuffix='_df4', how='inner')
# pd.merge()

df1
df2
pd.merge(df1, df2)
df4
df5
pd.merge(df4, df5, how='right')
df4['Grade'] = ['A', 'B', 'C', 'D']

df4
df5['Grade'] = ['A-', 'B+', 'C-', 'F']

df5
df_new = pd.merge(df4, df5, left_on='Name', right_on='Name', how='outer', suffixes=("_df4", "_df5"))

df_new.head()
pd.merge(df4, df5, left_index=True, right_index=True, how='outer')
# Groupby method

df6 = pd.DataFrame({'Name': ['Alice', 'Bob', 'Clare', 'Doug'],

                    'Gender': ['F', 'M', 'F', 'M'],

                    'Attendance': [9, 8, 7, 6]})

df6
df6groups = df6.groupby('Gender')

for name, group in df6groups:

    print(name, group.mean())

    print(group)
df6groups.mean()
df6groups.count()
df6
df_add = pd.DataFrame({

    'Name': ['Ed'],

    'Gender': ['M']

}, index=[5])

df_add

df6 = df6.append(df_add)

df6
# How to use the mean value of male student to fill the missing value?

# df6.groupby('Gender').mean()

# df6.loc[0, 'Attendence'] = 7.0
# Example of apply(): multiply attendence by 5

def multiply_by_5(x):

    return x * 5



df6['Attendance'].apply(multiply_by_5)
# def function5(x):

#     return x[0] * 5

# df6.apply(function5)
df6.apply(lambda x: x['Attendance'] * 5, axis=1)
df6
df6['Attendance'] = df6.groupby('Gender')['Attendance'].apply(lambda x: x.fillna(x.mean()))

df6