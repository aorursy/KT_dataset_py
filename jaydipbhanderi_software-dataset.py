# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df=pd.read_json("/kaggle/input/software/Software.json", lines="true")
df_meta=pd.read_json("/kaggle/input/meta-software/meta_Software.json", lines="true")
df_meta.head()
df.head()
print(len(df))
print(len(df_meta))
print("meta df columns:\n",df_meta.columns)
print("df columns:\n",df.columns)
len(df.asin.unique())
len(df_meta.asin.unique())
len(df_meta[df_meta.main_cat=="Software"])
print(df_meta.iloc[500,:])
print(df.iloc[1,:])
for col in df_meta.columns:
    print(col, sum(df_meta[col].isnull()))
for col in df.columns:
    print(col, sum(df[col].isnull()))
df[df.asin=="1603918310"]

