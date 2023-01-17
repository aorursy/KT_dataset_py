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
#import required libraies

import pandas as pd 

import numpy as np
#example 1 series creating with list

s = pd.Series(list('abcd')) # series will create will a b c d

s
#series creates based on ndarray

countries_arry = pd.array(['ind','aus','eng','Newz', 'sri'])

countries_code = pd.Series(countries_arry) #pass the ndarry to series function

countries_code
alphabetic_dict ={'a':1,'b':2,'c':3,'d':4} 

alphabetic_series= pd.Series(alphabetic_dict) # or alphabetic_dict =pd.Series(['a','b','c','d'],index=[1, 2, 3, 4])

alphabetic_series

scalar_series = pd.Series(2 , index=['a','b','c','d'])

scalar_series
#fetch first element fro series s

s[0]
#fetch first 3 elements in countries_code

#countries_code[0:3]

countries_code.iloc[0:3]
#fetch char for alphabetic_series using index or name 

alphabetic_series.loc['a']
