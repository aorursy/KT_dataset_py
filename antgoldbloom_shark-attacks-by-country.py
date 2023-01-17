# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



pd.set_option('max_rows',20)
df_shark = pd.read_csv('/kaggle/input/global-shark-attacks/GSAF5.csv',encoding="latin-1")

df_shark['Date'] = pd.to_datetime(df_shark['Date'],errors='coerce')
df_shark[(df_shark['Date'].dt.year >= 2010) & (df_shark['Date'].dt.year <= 2016)].groupby('Country')['Case Number'].count().sort_values(ascending=False)[:20]