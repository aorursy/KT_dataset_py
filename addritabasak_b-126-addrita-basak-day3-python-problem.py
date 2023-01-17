# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd
import numpy as np


cricket_data = {'name': ['Sourav', 'Dhoni', 'Kohli', 'Raina', 'Rohit', 'Yuvraj', 'Ashwin', 'Bhuvi', 'Dhawan', 'Unmesh'],
'score': [78, 102, 118, 22, 34, 20, np.nan, np.nan, 8, np.nan],
'matches': [1, 3, 2, 3, 2, 3, 1, 1, 2, 1]}

labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']

#1.Write a Pandas program to get the powers of an array values element-wise.

arr = np.array([1,2,3,4])

index = pd.Series(arr)

print("Original Array:")
print(index)
print('\n')

new_arr = index.pow(index)

print("New Array:")
print(new_arr)
print('\n\n')

#2.Write a Pandas program to create and display a DataFrame from a specified dictionary data which has the index labels. 

df = pd.DataFrame(cricket_data , index=labels)
print(df)
print('\n\n')

#3.Write a Pandas program to get the first 3 rows of a given DataFrame. 

print(df.head(3))
print('\n\n')

#4.Write a Pandas program to select the specified columns and rows from a given data frame.

new = df.iloc[[1,2,3],[1,2]]
print(new)
print('\n\n')

#5.Write a Pandas program to select the rows where the score is missing, i.e. is NaN.

null = df[df['score'].isnull()]
print(null)
print('\n\n')