import numpy as np
import pandas as pd

df = pd.read_csv('../input/bankcredit/bankakredi.csv',encoding='utf-8') 
df
df.describe().T
df.info()
Y = df["Risk"]
X = df.iloc[:,0:8]
X
Y
