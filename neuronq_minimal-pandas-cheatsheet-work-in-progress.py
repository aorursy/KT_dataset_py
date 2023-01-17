# try to be Python 3 compatible

from __future__ import division, print_function



# pretty printer

import pprint

pp = pprint.PrettyPrinter(indent=2).pprint
import numpy as np

import pandas as pd
# from array-like

s1 = pd.Series([1,3,5,np.nan,6,8])

s1
# from dict

d = {'a' : 0., 'b' : 1., 'c' : 2.}



print("from", d)

print("\nwith deduced index:")

pp(pd.Series(d))



index = ['b', 'c', 'd', 'a']

print("\nwith explicit index", index, ":")

pp(pd.Series(d, index))
# date_range is kind of like np.arange and np.linspace

# but for making dates indexes

# this gets six days starting from 2013-01-02

# (freq='D' means "day" and is actually the default)

pd.date_range('2013-01-02', periods=6, freq='D')
# create DF from array-like matrix

dates_index = pd.date_range('2013-01-02', periods=4)

pd.DataFrame(np.random.randn(4,5), index=dates_index, columns=list('ABCDE'))
# create DF from dict

df = pd.DataFrame({ 'A' : 1.,

               'B' : pd.Timestamp('20130102'),

               'C' : pd.Series(1,index=dates_index,dtype='float32'),

               'D' : np.array([3] * 4,dtype='int32'),

               'E' : pd.Categorical(["test","train","test","train"]),

               'F' : 'foo' },

             dates_index)

df
dfx = pd.DataFrame(np.random.randn(4,5), index=dates_index, columns=list('ABCDE'))

dfx
dfx.head(2) # default shows first 2
dfx.tail(3)
dfx.describe()
dfx['B'].iloc[1:3] = 1

dfx.mode()