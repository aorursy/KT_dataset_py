import numpy as np
import pandas as pd

df = pd.read_pickle('../input/progress1020.pkl')
#Drop Address, City, and Store Location columns and check they've been dropped
df=df.drop(['address','city','store_location'],axis=1)
list(df)