no_int=0
no_str=0

ls=['Subhrajyoti',37,19,13,'Maji','IEM','CSE']
for x in range(len(ls)):
    if type(ls[x])==str:
        no_str+=1
        print('String found at '+str(x))
    else:
        no_int+=1
        print('Integer found at '+str(x))
print('Number of interger: ',no_int)
print('Number of string: ',no_str)
a=0
b=0
list=["BINOD",17,32,"binod"]
for i in range(len(list)):
    if type(list[i])== str:
        a+=1
    else:
        b+=1
print("No. of string: ",a)
print("No. of integers: ",b)
no_int=0
no_str=0

ls=['Subhra','Jyoti',35,15,11,'Maji','IEM',45,'CSE']
for x in range(len(ls)):
    if type(ls[x])==str:
        no_str+=1
        
    else:
        no_int+=1
      
print('Number of interger: ',no_int)
print('Number of string: ',no_str)
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
