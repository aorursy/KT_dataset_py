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
range(4)
list(range(4))
for i in [[0,1],[1,3],[2,3]]:
    print(i)
for i in [[0,1],[1,3],[2,3]]:
    print(sum(i))
m = int(input())
for i in range(m):
    print( '*' * (i+1) )



print( 'a' *3 )
print( '*' + ' ' +'*')
m = input()
temp = len(m)//10
for i in range(temp):
    print( m[ i*10 : ((i+1)*10) ] )
    
if len(m) % 10 !=0 :
    print( m[temp*10:]  )
    




