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
n = int(input())

for i in range(1,10000000000):

    temp = i*n

    tem =str(i)

    t= tem[-1] + tem[0:-1]

    if(temp==int(t)):

        print(i)

        break

        
a='praveen'

b = a[-1]+ a[1:-1]

b