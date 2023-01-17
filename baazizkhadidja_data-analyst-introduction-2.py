import numpy as np

import pandas as pd



label = ['a', 'b', 'c']

liste = [10, 20, 30]

ar = np.array(liste)

ar

d = {'a':10, 'b':20, 'c':30}

d
pd.Series(liste,label)
note = {'et1':[18, 15, 14], 'et2':[11, 8, 9], 'et3':[2, 4, 6]}

note
pd.Series(note)
from numpy.random import randn



df = pd.DataFrame(randn(4,4), index = ['et1', 'et2', 'et3', 'et4'], columns = ['math', 'physiq', 'science', 'chimi'])

df
df[['math', 'chimi']]
df.drop('math', axis = 1, inplace = True)

df
df
df.drop('et1', inplace = True)
df
df[ (df['physiq']>0) | (df['science']>0)]
df['states'] = 'algeria tunis oran'.split()

df
outside = ['G1', 'G1', 'G1', 'G2', 'G2', 'G2']

inside = [1,2,3,1,2,3]



hier_index = list(zip(outside, inside))

hier_index
import pandas as pd

index_h = pd.MultiIndex.from_tuples(hier_index)

index_h
import numpy as np





df = pd.DataFrame(np.random.randn(6,2), index = index_h, columns = ['A', 'B'])

df
nom = ['doody', 'doody', 'doody', 'abdou', 'abdou', 'abdou']

note = ['ling', 'odio', 'pyth', 'dessin', 'math', 'science']



tableau = list(zip(nom, note))

tableau

tableau = pd.MultiIndex.from_tuples(tableau)

tableau

df = pd.DataFrame(np.random.randn(6,3), index = tableau, columns = ['S1', 'S2', 'S3'])

df
import pandas as pd

df = {'compagny':['GOOG', 'GOOG', 'MFST', 'MFST', 'FB','FB'], 

     'Person': ['Sam', 'Charle', 'Amy', 'Vanessa', 'Carl', 'Sara'],

     'Sales':[200, 120, 340, 124, 243, 350]}

df =pd.DataFrame(df)

df
df.groupby('compagny').describe().transpose()

right = {'c': ['c0','c1','c2','c3'],

         'd':['d0','d1','d2','d3'],

         'key':['k0','k4','k5','k6'] }



left ={'a': ['a0','a1','a2','a3'],

         'b':['b0','b1','b2','b3'],

         'key':['k0','k1','k2','k3'] }

left = pd.DataFrame(left)
right = pd.DataFrame(right)
pd.concat([left,right], axis = 0)
pd.merge(left,right, how = 'inner', on = 'key')