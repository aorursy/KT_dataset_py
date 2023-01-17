import numpy as np
import pandas as pd
url = "https://raw.githubusercontent.com/beoutbreakprepared/nCoV2019/master/latest_data/latestdata.csv"
df = pd.read_csv(url)
df.head()
df.tail()
df.info()
for c in df.columns:
    print(c  + '\n')
    print(df[c].unique())
    print("\n")
