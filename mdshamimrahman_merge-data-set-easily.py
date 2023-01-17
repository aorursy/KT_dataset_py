
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


df=pd.read_csv('../input/heartbeat-sounds/set_a.csv')
df.head(5)
df2=pd.read_csv('../input/heartbeat-sounds/set_a_timing.csv')
df2.head(5)
df3=pd.read_csv('../input/heartbeat-sounds/set_b.csv')
df3.head(5)
ndf=pd.merge(df,df2,on='fname',how='outer',indicator=True)
## We use how becase the data set is not same as previous (Example: if previous one is 10 information thn second 
##one is 11 information)
ndf.head(5)

newdf=pd.merge(ndf,df3,on=['dataset','fname','label'],how='outer',indicator='true')
newdf.head(5)
