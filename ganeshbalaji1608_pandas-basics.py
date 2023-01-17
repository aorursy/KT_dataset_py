import pandas as pd

import numpy as np
labels = [1,2,3]

data = ['a','b','c']

d = pd.Series(data,labels)
dic = {'ganesh' : 76, 'Ragul' : 89, 'Babu' : 90}
ds = pd.Series(dic)
dic2 = {'ganesh' : 80, 'Ragul' : 74, "dinesh" : 80}
ds2 = pd.Series(dic2)
df = pd.DataFrame(np.random.randn(4,4), index =  ['a','b','c','d'], columns =  ['A','B','C','D'])
df
df.iloc[1]
dic = {"A" : ["Ganesh","Dinesh", "Rahul"], "B" : ["Babu", "Gopi", "Mahesh"], "C" : ["Dinu", "viru", "lokesh"]}
dffromdic = pd.DataFrame(dic, index = ["Upper", "Middle", "Lower"])
dffromdic
df2 = dffromdic.copy()
df2.iloc[1:][['B']]
df2.drop('C',axis = 1)
df
df[(df['A']>0) & (df['B'] <0)]
df['E'] = np.random.randn(4)
df.reset_index(inplace = True)
df
df.set_index('index', inplace = True)
df
outside = ['A', 'A', 'A', 'B', 'B', 'B']

inside = ['loop', 'hole', 'hole', 'hole', 'loop', 'loop']

index = list(zip(outside,inside))
index
hier_index = pd.MultiIndex.from_tuples(index)
dhier = pd.DataFrame(np.random.randn(6,2), hier_index, columns = ['first', 'second'])
dhier
dfmiss = pd.DataFrame({'A' : [1,np.nan,2], 'B' : [1,2,3], 'C' : [np.nan,np.nan,3]})
dfmiss
dfmiss.dropna(axis = 1, thresh=2)
dfmiss['C'].fillna(df['C'].mean, inplace = True)
dfmiss
dataf = pd.DataFrame