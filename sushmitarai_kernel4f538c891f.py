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
import pandas as pd



#

pd.DataFrame({'Yes': [50, 21], 'No': [131, 2]})



pd.DataFrame({'Bob': ['I liked it.', 'It was awful.'], 'Sue': ['Pretty good.', 'Bland.']})



#assigning row name

pd.DataFrame({'Bob': ['I liked it.', 'It was awful.'], 

              'Sue': ['Pretty good.', 'Bland.']},

             index=['Product A', 'Product B'])



#series

pd.Series([1, 2, 3, 4, 5])

#assigning column values to series

pd.Series([30, 35, 40], index=['2015 Sales', '2016 Sales', '2017 Sales'], name='Product A')



#importing csv, excel and txt file

import os

os.getcwd()



print(os.listdir("../input"))

data_1 = pd.read_csv("../input/data-temp-file/data_02.csv")

print(data_1)

data_1.shape

data_1.head()



#importing txt file (tab delimited)

data_02 = pd.read_csv("../input/data-temp-file/Data_temp_1.txt")

print(data_02)

data_02.head()



#importing excel file (not working)

data_03 = pd.read_xlsx("../input/data-temp-file/Data_temp.xlsx")

print(data_02)


