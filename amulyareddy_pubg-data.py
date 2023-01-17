# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import pandas as pd

df = pd.read_csv("../input/deaths/kill_match_stats_final_0.csv")

df.head()
df2 = pd.read_csv("../input/aggregate/agg_match_stats_0.csv")

df2.head()
import matplotlib.pyplot as plt



df.dtypes
df_copy = df.copy()
df_copy = df_copy.drop(['match_id'], axis=1)
df_copy.head()

df_c = df_copy.groupby(['killer_name'])

df_c.first()
len(df_copy)//len(df_copy['killer_name'].unique())
df_c = df_copy.groupby(['killed_by'])

df_c.first()