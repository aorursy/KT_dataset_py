import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_csv("/kaggle/input/world-happiness/2019.csv")



data.columns   
data.columns = [each.lower() for each in data.columns]   # lowercases each column by for loop

print(data.columns)                                      # prints the columns
data.columns = [each.upper() for each in data.columns]  # uppercases all columns

print(data.columns)
for each in data.columns:    # Cycyles each column in data

    each.lower()             # lowercased each column

    print(each)              # prints each column
print(data.columns[0]) # column index 0 ---> overal rank

print(data.columns[1]) # column index 0 ---> country or region
list_columns = list(data.columns)   # Convert data.columns to list and assign it to list_columns 

i=0

for each in data.columns:           # Cycyles each column in data.columns

    list_columns[i]= each.lower()   # Lowercases each column and assigns them to indexes of list_columns 

    i = i + 1                       

    data.columns = list_columns     # Equalizes data.columns with values of list_columns 

print(data.columns)                 # Prints columns of data
data.columns = [each.upper() for each in data.columns]   # Uppercases the columns

print(data.columns)
list_columns = list(data.columns)    # Converts data.columns to list and assing to list_columns

a = map(lambda x:x.lower(), list_columns)    # a is a map object includes lowercased columns

b = list(a)  # b is a list which is converted from a

data.columns = b   # values of list b assigned to data.coloumns

print(data.columns)



# Below line will also works in place of 3 lines of coding

# data.columns = list(map(lambda x: x.lower(), list(data.columns)))
lis1=list(data.columns)        # Converting data.column to list of lis_columns

i=0

for each in data.columns:      # Cycles each column in data

  lis1[i]=len(each.split())    # Splits each column and assign the length of each split to list_columns 

  i=i+1 

n = max(lis1)                  # Assignes max split word count in any column in data to "n" 

print(lis1)                    # [2, 3, 1, 3, 2, 3, 5, 1, 3]

print(n)
lis2 = list(data.columns)      # Conveting data.columns to list and assigning it to lis2

i=0

for each in data.columns:

  

  if len(each.split()) > n-1:

    lis2[i] = each.split()[0]+"_"+ each.split()[1]+"_"+each.split()[2]+"_"+each.split()[3]+"_"+each.split()[4]

    i=i+1

  elif len(each.split()) > n-2:

    lis2[i] = each.split()[0]+"_"+ each.split()[1]+"_"+each.split()[2]+"_"+each.split()[3]

    i=i+1

  elif len(each.split()) > n-3:

    lis2[i] = each.split()[0]+"_"+ each.split()[1]+"_"+each.split()[2]

    i=i+1

  elif len(each.split()) > n-4:

    lis2[i] = each.split()[0]+"_"+ each.split()[1]

    i=i+1

  else:

    lis2[i] = each

    i=i+1

data.columns = lis2

print(data.columns)