import pandas as pd
import numpy as np
from numpy.random import randn
np.random.seed(1000)
random_test = randn(2,2)
print(random_test)
df = pd.DataFrame(randn(5,4),index='A B C D E'.split(),columns='W X Y Z'.split())
df_2 = pd.DataFrame(randn(3,2), index='1 2 3'.split(), columns='Pepaya Mangga'.split())
df
df_2
tes = 'Orthogonalized Plane Wave'
tes_split = tes.split()
print(tes_split)
print(type(df[['W', 'Y']]))
print(df_2['Pepaya'])
# Pass a list of column names
df[['W','Z']]
# SQL Syntax (NOT RECOMMENDED!)
df.W
type(df['W'])
df_2['Smoothies'] = df_2['Pepaya'] + df_2['Mangga']
df_2
df.drop('new',axis=1)
df_2.drop('Smoothies', axis=1)
df_2.drop('2', axis=0)
# Not inplace unless specified!
df_2
df_2.drop('Pepaya',axis=1,inplace=True)
df_2
df_2
df.drop('E',axis=0)
df.loc['A']
df.iloc[2]
df.loc['B','Y']
df_2.loc['2','Smoothies']
df.loc[['A','B'],['W','Y']]
df_2
df_2.loc[['1','2'],['Mangga','Smoothies']]
df
df>0
df[df>0]
df[df['W']>0]
df[df['W']>0]['Y']
df[df['W']>0][['Y','X']]
df[(df['W']>0) & (df['Y'] > 1)]
df
# Reset to default 0,1...n index
df.reset_index()
newind = 'CA NY WY OR CO'.split()
df['States'] = newind
df
df.set_index('States')
df
df.set_index('States',inplace=True)
df
# Index Levels
outside = ['G1','G1','G1','G2','G2','G2']
inside = [1,2,3,1,2,3]
hier_index = list(zip(outside,inside))
hier_index = pd.MultiIndex.from_tuples(hier_index)
hier_index
df = pd.DataFrame(np.random.randn(6,2),index=hier_index,columns=['A','B'])
df
df.loc['G1']
df.loc['G1'].loc[1]
df.index.names
df.index.names = ['Group','Num']
df
df.xs('G1')
df.xs(['G1',1])
df.xs(1,level='Num')