# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/commodity_trade_statistics_data.csv", na_values=["No Quantity",0.0,''],sep=',')
df.shape
df.year.describe()
df.isnull().sum()
df["year"] = df["year"]+1  # we want the date to be the first day of the next year
df["year"] = pd.to_datetime(df["year"],format="%Y")
df.head()
df.drop(["weight_kg","quantity_name"],axis=1,inplace=True)

df = df.dropna(how='any').reset_index(drop=True)
df.shape
df.flow.value_counts()
df[df.flow=="Re-Export"].head()
df.to_csv("commodity_trade_stats_global.csv.gz",index=False,compression="gzip")