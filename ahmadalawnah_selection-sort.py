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
#selection sort algorithm

list = [4, 3, 2, 1, 7, 9, 2, 1]

minIndex = 0

for i in range (0,len(list)):

    minIndex = i

    for j in range(i+1,len(list)):

        if list[j] < list[minIndex]:

            minIndex = j;

    if i != minIndex:

        list[i], list[minIndex] = list[minIndex], list[i]

print(list)