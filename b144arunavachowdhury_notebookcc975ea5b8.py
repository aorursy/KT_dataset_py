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
data=pd.read_csv('/kaggle/input/titanic/train_and_test2.csv')

print(data)
list_a=["Python","C","C++","Java"]

tuple_a=("Python","C","C++","Java")

dict_a={1:"Python",2:"C",3:"C++",4:"Java"}

#list is mutable in nature and the slicing operator can be used on it

list_a[1] = "Kotlin"

print(list_a)

list_a.insert(4,"PHP")

print(list_a)

list_a.pop(3)

print(list_a)

print(list_a[0:2])

#tuples are not mutable in nature, although the slicing operator can be used 

#unlike lists,no changes can be made to the tuple

print(tuple_a)

print(tuple_a[0:2])

#tuple_a[2]="React_Native",this would lead to an error

#dictionaries have two parts, i.e keys and values, for eg, dict ={key1:"value1",key2:"value2"}

print(dict_a)

#in dictionary the "keys" are like tuples and are immutable while "values" are like lists and mutable

#to change the values, we might reference the "values" through the "keys"

dict_a[1]="Kotlin"

print(dict_a)

#removing a key and its value

del dict_a[1]

print(dict_a)

#adding a new key and its value

dict_a[1]="React_Native"

print(dict_a)