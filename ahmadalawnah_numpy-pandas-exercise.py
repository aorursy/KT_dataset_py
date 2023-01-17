# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#question 3

arr = np.array([2,3,4,5,6,7,8,9,10])

print("Before:")

print(arr)

arr = arr.reshape((3,3))

print("After:")

print(arr)
#question 73

arr = np.array([1,2,3,4,5,6,7,8,9,10,11,12]).reshape((3,4))

print("Before:")

print(arr)

arr = arr*3

print("After:")

print(arr)
#pandas DataFrame 3 and 4

exam_data = {'name': ['Anastasia', 'Dima', 'Katherine',

'James', 'Emily', 'Michael', 'Matthew', 'Laura', 'Kevin',

'Jonas'],

'score': [12.5, 9, 16.5, np.nan, 9, 20, 14.5, np.nan, 8,

19],

'attempts': [1, 3, 2, 3, 2, 3, 1, 1, 2, 1],

'qualify': ['yes', 'no', 'yes', 'no', 'no', 'yes', 'yes',

'no', 'no', 'yes']}



data = pd.DataFrame(exam_data)

print("Original data:")

print(data)



newData = data[['name', 'score']]

newData = newData.head(3)

print("Modified:")

print(newData)