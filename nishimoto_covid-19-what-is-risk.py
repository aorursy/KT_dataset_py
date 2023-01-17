import os

import numpy as np

import pandas as pd
df = pd.read_csv("/kaggle/input/CORD-19-research-challenge/metadata.csv")



# before 2020, maybe publication about COVID-2019 is not out.

df_2020 = df.query("'2020' in publish_time")



# I think Confidence Interval (CI) is used epidemiological evaluation

df_2020_ci = df_2020.loc[df_2020["abstract"].str.contains("CI").fillna(False), :]
print(df_2020_ci.shape)

df_2020_ci.head()