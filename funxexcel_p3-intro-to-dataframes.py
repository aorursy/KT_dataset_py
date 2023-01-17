import pandas as pd
import numpy as np
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
df = pd.read_excel('3_NEWS_Sales.xlsx',sheet_name='NEWS')
df
df.set_index('Month', inplace = True)
df
df['North']
# Pass a list of column names
df[ ['North','South'] ]
df['Jan']
df.loc['Jan']
type(df['North'])
type(df.loc['Jan'])
df['North_South'] = df['North'] + df['South']
df
df.drop('North_South',axis=1)
# Not inplace unless specified!
df
df.drop('North_South',axis=1,inplace=True)
df
df.drop('Jan',axis=0)
df.loc['Jan']
df.iloc[0]
df.loc['Mar','South']
df.loc[['Feb','Mar'],['East','West']]
df
df>5000
df[df>5000]
df[df['North']>8000]
df[df['North']>8000]['East']
df[df['North']>8000][['East','West']]
df[(df['North']>8000) & (df['East'] > 5000)]
df
# Reset to default 0,1...n index
df.reset_index()
newind = 'Jan19 Feb19 Mar19 Apr19 May19 Jun19 Jul19 Aug19 Sep19 Oct19 Nov19 Dec19'.split()
df['Year_19'] = newind
df
df.set_index('Year_19')
df
df.set_index('Year_19',inplace=True)
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