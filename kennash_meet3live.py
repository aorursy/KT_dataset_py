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
a = 1
print(a)
a = "Hello"
print("Hello")
a = (1<=1)
print(a)
'''Boolean, Integer, String, Float'''
a = 1.1
print(a)
type(7.0)
type('12')
print(12+12)
print('12'+'12')
print('hello'+' world')
print('multiply'*3)
print('multiply'*"multiply")
print(['steven', 'jeff', 'jeph'])
a = ['steven', 'jeff', 'ken']
print(a)
print(a[1])
print(a[0])
names= ['steven', 'jeff', 'ken']
names
names[:]
names[1:2]
names[-1]
fruits = ["apple", "banana", "pineapple", "guava", "peach", "kiwi"]
fruits
fruits[:]
fruits[0:7]
fruits[0:7:1]
fruits[::-1]
a = [1, 2, 3][1:]
len(a)
print(a)
fruits[1:]
print(["a", "b","c"][1:3])
def minimize(a, b): 
    if (a<b):
        return a
    elif (a==b):
        return "They are equal"
    else:
        return b
minimize(10, 3)