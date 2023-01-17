import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
df = pd.DataFrame({'col1':[1,2,3,4],'col2':[444,555,666,444],'col3':['abc','def','ghi','xyz']})

df.head()
df['col2'].unique()
df['col2'].nunique()
df['col2'].value_counts()
df[(df['col1']>2) & (df['col2'] == 444)]
def times2(x):

    return x*2
df['col1'].apply(times2)
df['col3'].apply(len)
df['col1'].apply(lambda x: x*2)
df.columns
df.index
df.sort_values('col2')
df.isnull()
data = {'A':['foo','foo','foo','bar','bar','bar'],

     'B':['one','one','two','two','one','one'],

       'C':['x','y','x','y','x','y'],

       'D':[1,3,2,5,4,1]}



df = pd.DataFrame(data)

df
df.pivot_table(values='D', index=['A', 'B'], columns=['C'])