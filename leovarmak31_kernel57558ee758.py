import pandas as pd
df = pd.read_csv("../input/index.csv")
        
        

df.head()
df.tail()
df.columns
df.iloc[::2]
df.iloc[:,:2]
df.iloc[:,2:3]
df.iloc[:,-2:]
df.iloc[-10:,:2]



