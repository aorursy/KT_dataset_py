# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
myList = [1,2,3,4,5,6,7,8,9,10]



for i in myList:

    if i % 2 == 0:

        myList.remove(i)

        

print(myList)
first_list = [1, 2, 3,4, 5]

second_list = [6, 7,8, 9,10]



for i in first_list:

    if i % 2 == 0:

        first_list.remove(i)

for x in second_list:

     if x % 2 != 0:

            second_list.remove(x)

                

resultList= list(set(first_list) | set(second_list))

    

print(resultList)