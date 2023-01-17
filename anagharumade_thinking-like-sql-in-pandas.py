import pandas as pd
import numpy as np
df = pd.DataFrame({'A' : ['foo', 'bar', 'foo', 'bar'],
                   	'B' : ['one', 'one', 'two', 'three'],
                   	'C' : [1, 2, 3, 4],
                   	'D' : [10, 20, 30, 40]})
df
df[df['A']=='foo']
df.sort_values(['A', 'B'])
df.sort_values(['A', 'C'], ascending=[False, True])
df.groupby('A').size()
df.groupby('A').count()
df.groupby('A').sum()
df1 = pd.DataFrame({'A': [10, 20, 30, 50], 'B': ['TEN', 'TWENTY', 'THIRTY', 'FIFTY']})
df1
pd.merge(df, df1, left_on='D', right_on='A')
pd.merge(df, df1, how='left', left_on='D', right_on='A')
pd.merge(df, df1, how='right', left_on='D', right_on='A')
pd.merge(df, df1, how='outer', left_on='D', right_on='A')