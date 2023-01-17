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

for elem in range(5):
	print(elem, end=' ')
print()


for elem in range(1,6):
	print(elem, end=' ')
print()


for elem in range(5,-1,-1):
	print('Countdown:',elem)

for char in 'string':
	print(char,end=' ')
print()
for tup in (1,3,5):
	print(tup)
for val in ['hey', 'hi', 'hello']:
	print(val)
greek={'alpha':1,'beta':2,'gamma':3}
for key in greek:
	if key=='beta':
		continue
	print(key,greek[key])
for outer in range(2,10):
	for inner in range(2,outer):
		if not outer%inner:
			print(outer,'=',inner,'*',int(outer/inner))
			break
		else:
			print(outer,'is prime') 