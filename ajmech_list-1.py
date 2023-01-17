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
# ipAddress = input("Please enter an IP address")
# print(ipAddress.count("a"))
parrot_list = ["non pinin", "no more", "a stiff", "bereft of live"]
parrot_list.append("A Norwegian Blue")
for i in parrot_list:
    print("Parrot is " + i)

print()

print(parrot_list)

even = [2, 4, 6, 8]
odd = [1, 3, 5, 7, 9]
numbers = even + odd # Concatenates the lists
print(numbers)

numbers.sort() # This does not return the sorted list, only works in original object

print(numbers)

numbers = even + odd
print(sorted(numbers)) # sorted(list) returns the sorted list
