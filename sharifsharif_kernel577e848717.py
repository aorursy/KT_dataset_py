# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)





myArray = np.arange(2,11).reshape(3,3)

print(myArray)
import numpy as np # linear algebra





myArray = np.arange(1,13).reshape(3,4)



print(myArray**3)
import pandas as pd  

import numpy as np

exam_data = {'name': ['Anastasia', 'Dima', 'Katherine',

'James', 'Emily', 'Michael', 'Matthew', 'Laura', 'Kevin',

'Jonas'],

'score': [12.5, 9, 16.5, np.nan, 9, 20, 14.5, np.nan, 8,

19],

'attempts': [1, 3, 2, 3, 2, 3, 1, 1, 2, 1],

'qualify': ['yes', 'no', 'yes', 'no', 'no', 'yes', 'yes',

'no', 'no', 'yes']}

labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']



print("To Print The 3 Columns of the Dictionary ")

print("---------------------------------------")

print()

myDictionary = pd.DataFrame(exam_data , labels)

print(myDictionary.iloc[:3])     # this is to get the first 3 Columns of 



print("-----------------------")

print("To Print The Names and Scores of each Student ")

print()

print(myDictionary[['name', 'score']])
