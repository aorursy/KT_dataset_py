import pandas as pd
import numpy as np
num_data = pd.Series([5,6,7,3,0,np.nan])
num_data.isna()
str_data = pd.Series(['Trust','Commitment','Love',None])
str_data.isna()
str_data1 = pd.Series(['Eat',np.nan,'Pray','Love','Sleep'])
str_data.isnull()
str_data.isna()
arr = np.array([3.0,4.0,5.0,np.nan,6.0])
np.isnan(arr)
arr1 = np.array(['Fire','Ice',None,'Blood'])
None in arr1
t = ('Ed','Zed','Ted',None)
None in t

l = [1,2,3,4,4,np.nan]
np.nan in l
x = (['I','have','only',np.nan,1,'bestie',None])
(np.nan and None) in x
(np.nan or None) in x