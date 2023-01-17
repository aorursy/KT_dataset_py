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
df = pd.read_csv('../input/startup_funding.csv')
df.info()
df["AmountInUSD"] = df.AmountInUSD.fillna(0)
df["AmountInUSD"] = df["AmountInUSD"].str.replace(',','').fillna(0)
df["AmountInUSD"] = pd.to_numeric(df["AmountInUSD"])
df.sort_values(['AmountInUSD'],ascending=False)[0:5]
df.Date = df.Date.str.replace('.','/')
df.Date = df.Date.str.replace('//','/')
df.Date = pd.to_datetime(df.Date, format='%d/%m/%Y')
df[df.Date >= ' 2017-01-01']['AmountInUSD'].sum()/1000000
