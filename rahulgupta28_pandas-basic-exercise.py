import numpy as np

import pandas as pd
#Creating Series

lables = ['a','b','c','d','e']

my_list = [11,21,31,41,51]

l1 = ['aa','bb','cc']

# pd.Series(data=None , index = None , dtype = None , copy = False , fastpath = False) -> by default arguments for series data structure

pd.Series(data=my_list)
pd.Series(data=my_list,index=lables)
pd.Series(data=my_list,index=l1)
#chnage the dtyoe

pd.Series(data = my_list,index=lables,dtype=float)
S1 = pd.Series([1,2,3,4],index=['USA','GER','USSR','India'])

S2 = pd.Series([1,2,3,4],index=['USA','GER','Italy','India'])

S1+S2
S1-S2
S1*S2
S1/S2
S1%S2
#pd.DataFrame(data = None , index = None , columns = None , dtpye = None , copy = False)



d={'col':[1,2],'col2':[3,4]}

df=pd.DataFrame(data = d)

print(df.dtypes)

df
df = pd.DataFrame(data = d , dtype = np.int8)

print(df.dtypes)
from numpy.random import randn

np.random.seed(101)

df = pd.DataFrame(randn(5,4),index = 'A B C D E'.split(),columns = 'w x y z'.split())

df