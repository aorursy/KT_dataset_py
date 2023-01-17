import numpy as np
import pandas as pd
labels = ['a', 'b', 'c']
my_data=[10,20,30]
arr = np.array(my_data)
d = {'a':10, 'b':20, 'c':30}
pd.Series(data=my_data)  #make a serie
pd.Series(data=my_data,index=labels) #show label list as index of the Series
pd.Series(d)
pd.Series(arr, labels)
ser1 = pd.Series([1,2,3,4],['USA','Germany','USSR','Japan'])
ser1
ser1['USA']
ser3 = pd.Series(data=labels)
ser3
ser3[0] 
from numpy.random import randn
np.random.seed(101)

df = pd.DataFrame(randn(5,4),['A','B','C','D','E'],['W','X','Y','Z'])   

df 
df['W']