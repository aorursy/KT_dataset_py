# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
age=0
if age:
	print('False condition will not execute')
	print('So these statements will not print')
age=1
if age:
	print('True condition will execute')
	print('So these statements will print')
age=17
if age>18:
	print('You are eligible to vote')
else:
	print('Not eligible to vote')
score=91
print('the grade is:',end='')
if score<60:
	print('E')
elif 60<=score<70:
	print('D')
elif 70<=score<80:
	print('C')
elif 80<=score<90:
	print('B')
elif 90<=score<100:
	print('A')
else: 
	print('Invalid')
