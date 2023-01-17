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
#so that we dont have to write pd.series and pd.dataframe again and again
from pandas import Series, DataFrame
# a series is like an array in numpy except it has data labels so it is indexed
obj=Series([3,6,9,12])
obj
obj.values
obj.index
ww2_cas=Series([8700000,4300000,3000000,2100000,400000],index=['USSR','Germany','China','Japan','USA'])
ww2_cas
ww2_cas['USA']
#check which countries had casualties greater than 4 million
ww2_cas[ww2_cas>4000000]
'USSR' in ww2_cas
ww2_dict=ww2_cas.to_dict()
ww2_dict
ww2_series=Series(ww2_dict)
ww2_series
#we can feed dictioonaries into series and now we have a series again
countries=['China','Germany','Japan','USA','USSR','Argentina']
obj2=Series(ww2_dict,index=countries)
obj2
pd.isnull(obj2)
pd.notnull(obj2)
ww2_series
#we can add series to series and it will automatically alighn indexes
obj2
ww2_series + obj2 
obj2.name='World War 2 Casualities'
obj2
obj2.index.name='Countries'
obj2
