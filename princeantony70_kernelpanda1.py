import numpy as np

import pandas as pd

label=['a','b','c']

my_data =[10,20,30]

arr = np.array(my_data)

d ={'ax':10,'by':20,'cz':30}



pd.Series(data = my_data)
pd.Series(data = my_data,index=label)
pd.Series(data = arr,index = label)
pd.Series(d)
pd.Series(label)
ser1 = pd.Series(['TN','KL','AN'],label)

ser1
serr1 =pd.Series([1,2,3],['USA','USSR','JAPAN'],)

serr1
serr2 =pd.Series([1,2,3],['SA','USSR','JAPAN'],)

serr2
serr1+serr2