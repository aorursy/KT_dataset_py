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
num = 15
if (num < 10):
    print("Value of num is less than 10")
else:
    print("Value of num is greater than 10")
a = 11
b = 20
c = 30
if (a > b and a > c):
    print('a is greatest number')
elif(b > a and b > c):
    print('b is greatest number')
else:
    print("c is greatest number")
num = 11
if (num < 10):
    print('Value of num is less than 10')
elif(num > 10):
    print('Value of num is greater than 10')
else:
    print('Value of num is equal to 10')
age = 29
qualification = 'UG'
maritial_status = 'single'

if(age<35 and qualification == 'PG' and maritial_status == 'married'):
    print('you qualify for visa')
elif(age>35):
    print('Age bar test fail! not applicable for visa')
elif(qualification=='UG'):
    print('You are underqualified for visa')
elif(maritial_status != 'married'):
    print('You have failed the maritial qualification, not eligible for visa')
age = 32
qualification = 'PG'
maritial_status = 'married'

if(age<35 and qualification == 'PG' and maritial_status == 'married'):
    print('you qualify for Australian immigration')
elif(age>35):
    print('Age bar test fail! not applicable for Australian immigration')
elif(qualification=='UG'):
    print('You are underqualified to immigrate to Australia')
elif(maritial_status != 'married'):
    print('You have failed the maritial qualification, no eligible for immigration')
