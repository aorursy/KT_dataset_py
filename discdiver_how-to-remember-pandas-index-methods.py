import pandas as pd

import numpy as np
df = pd.DataFrame(dict(a=[1,2,3,4], b=[2,5,6,4]))

df
df2 = pd.DataFrame(dict(a=[1,2,3,4], b=[2,5,6,4]), index = [1,2,5,6])

df2
df3 = df2.set_index("a")

df3
df3.index = [2,3,4,5]

df3
df4 = df3.reset_index()

df4
df5 = df3.reset_index(drop=True)

df5
df6 = df5.reindex([2,3,1,0])

df6
df7 = df5.reindex([2,3,1,0,6])

df7