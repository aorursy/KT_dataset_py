import pandas as pd
# Get the powers of an array values element-wise
df = pd.DataFrame({'A':[78,85,96,80,86], 'B':[84,94,89,83,86],'C':[86,97,96,72,83]});
print(df)
# Create and display a Dataframe from a dictionary data which has the index labels
import numpy as np
exam_data  = {'name': ['Virat', 'Rohit', 'Dhoni', 'Shikhar', 'Jadeja', 'Ashwin', 'Rahane', 'Rahul', 'Rishabh', 'Iyer'],
        'score': [12.5, 9, 16.5, np.nan, 9, 20, 14.5, np.nan, 8, 19],
        'attempts': [1, 3, 2, 3, 2, 3, 1, 1, 2, 1],
        'qualify': ['yes', 'no', 'yes', 'no', 'no', 'yes', 'yes', 'no', 'no', 'yes']}
labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']

df = pd.DataFrame(exam_data , index=labels)
print(df)
# Get the first 3 rows of a given Dataframe _ here we use the above dataframe df
print("First three rows of the data frame:")
print(df.iloc[:3])
# Select the specified columns and rows from a given Dataframe _ here we use the above dataframe df
print("Select specific columns and rows:")
print(df.iloc[[1, 2, 5, 9], [0, 1, 3]]) 
# we choose the rows indexed 1,2,5,9 and columns with index 0,1,3
# Select the rows where the score is missing, i.e. is NaN. from dataframe df
print("Rows where score is missing:")
print(df[df['score'].isnull()])