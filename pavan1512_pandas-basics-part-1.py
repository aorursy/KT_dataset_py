import pandas as pd
# create list, tuple and dict

l1 = [23, 43, 56, 78, 87, 54]

t1 = 23, 43, 56, 78, 87, 54

d1 = {'Emp ID': 1001, 'Emp Name': 'Pavan', 'Dept': 'SAP', 'Salary': 30000}
# Create Pandas Series from above

pd.Series(l1)
pd.Series(t1)
pd.Series(d1)
# Assigning name to the column and custom index

s1 = pd.Series(l1, name = 'Age', index = range(2, 8))

s1
# Accessing the elements

s1[s1 > 60]
# .iloc: 

    # in case we want to get the values from ds using default indexes [int]

    # in case range is given for slicing; range will be low till end -1

    

# .loc: 

    # in case we want to get the values from ds using user defined indexes [Numeric/str]

    # in case range is given for slicing; range will be low till end
s1
s1.loc[2:4]
# We can access using iloc even after assigning custom index

s1.iloc[2:4]

# Third element in s1 = 56, and fourth element in s1 = 78
# indexing on multiple conditions: get the elements where vlaue is greater than 50 and less than 90

s1[(s1 > 50) & (s1 < 90)]
s1.loc[(s1 > 50) & (s1 < 90)]
s1.iloc[list((s1 > 50) & (s1 < 90))]
# third element from the series (We need to use custom index value)

s1.loc[4]
s1.iloc[2] # We need to use default index value
l1
t1
# create a dataframe from the list/tuple

pd.DataFrame(t1)
d1 = {'EmpID': [1001, 1002, 1003, 1004],

         'Emp Name': ['Pavan', 'Kumar', 'Reddy', 'Idula'],

             'Dept': ['HR', 'IT', 'IT', 'Finance'],

                 'Salary': [10000, 12000, 14000, 13000]}

d1
# Create a Dataframe from dictinary

pd.DataFrame(d1, index = range(0, 4))
# Create a Dataframe from pandas series

s1 = pd.Series([1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008], name = 'EmpId', dtype = int)

s2 = pd.Series(['John', 'Mac', 'Raj', 'Tim', 'Lee', 'Sam', 'Kim', 'Joe'], name = 'EmpName', dtype = object)

s3 = pd.Series(['Finance', 'HR', 'IT', 'IT', 'IT', 'IT', 'Finance', 'HR'], name = 'Dept', dtype = object)

s4 = pd.Series([10000, 12000, 10000, 12000, 14000, 13000, 10000, 18000], name = 'Salary', dtype = int)
df = pd.DataFrame([s1, s2, s3, s4])

df
# We need to apply transpose for above

df1 = df.T

df1
import numpy as np
# Create a Dataframe from numpy arrays

a1 = np.full((3,3), 6)

a1
df = pd.DataFrame(a1, columns = ['ABC', 'XXX', 'XYZ'])

df