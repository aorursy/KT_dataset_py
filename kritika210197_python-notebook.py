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
counter=3
while counter>0:
	print("counting down:", counter)
	counter-=1
while counter>0:
	print('Never executes this statement')
	print('condition is false')    
while 1:
	print('Execute at least once')
	if not counter:
		break

names= ['Tom','Ellen']
while names:
	print(names.pop(), 'is going')


results=[1,0,1]
processed=0
passed=0
while results:
	processed+=1
	result=results.pop()
	if not result:
		continue
	passed+=1
else:
    print('Pocessed:',processed,'Passed:',passed)
