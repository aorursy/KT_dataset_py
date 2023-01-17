import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os 
print(os.listdir("../input"))

df=pd.read_json("../input/searches.json",lines=True)
df.head(5)
records=len(df.index)
count=0

for index , row in df.iterrows() :
    if row['search_count'] is 0 :
        count=count+1
        
if count >= records/2 :
    print("more users use search features in new design(B)")
    
df1=df.iloc[:,[3,2]]
df1.head()

df1.sort_values('search_count',ascending=False)
scount=0
for index,row in df.iterrows():
    if row['search_count'] >3 :
        scount=scount+1

x = records-scount
if scount>=x:
    print("users search more often in the new design (B)")
else:
    print("users dont search more often in the new design(B)")

