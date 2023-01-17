



import numpy as np

import pandas as pd 









import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



print("Hello All")
import pandas as pd

deliveries = pd.read_csv("../input/ipldata/deliveries.csv")

df_matches = pd.read_csv("../input/ipldata/matches.csv")



df_matches.head(3)
df_matches.head(3).transpose()
list(df_matches.columns)
df_matches.shape
df_matches.info()
df_matches[0:1]