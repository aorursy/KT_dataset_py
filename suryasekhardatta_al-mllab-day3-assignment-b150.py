#AI & ML Lab - Day 3 Assignment

import pandas as pd
import numpy as np


marks_data = {'name': ['John', 'Liam', 'Emma', 'Noah', 'Oliva', 'Logan', 'Lucas', 'Mason', 'James', 'Umesh'],
'marks': [75, 100, 36, 57, 34, 78, np.nan, np.nan, 82, np.nan],
'points': [8, 10, 4, 6, 4, 8, np.nan, np.nan, 9, np.nan]}

labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']

#1) Write a Pandas program to get the powers of an array values element-wise.

arr = np.array([1,2,3,4])

index = pd.Series(arr)

print("Original Array:")
print(index)
print('\n')

new_arr = index.pow(index)

print("New Array:")
print(new_arr)
print('\n\n')

#2) Write a Pandas program to create and display a DataFrame from a specified dictionary data which has the index labels. 

df = pd.DataFrame(marks_data , index=labels)
print(df)
print('\n\n')

#3) Write a Pandas program to get the first 3 rows of a given DataFrame. 

print(df.head(3))
print('\n\n')

#4) Write a Pandas program to select the specified columns and rows from a given data frame.

new = df.iloc[[1,2,3],[1,2]]
print(new)
print('\n\n')

#5) Write a Pandas program to select the rows where the score is missing, i.e. is NaN.

null = df[df['marks'].isnull()]
print(null)
print('\n\n')
