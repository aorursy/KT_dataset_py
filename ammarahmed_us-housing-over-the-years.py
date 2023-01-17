# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
df=pd.read_csv('../input/family-households-with-married-couples.csv')
# Any results you write to the current directory are saved as output.
df["date"]=pd.to_datetime(df["date"])
df2=df.sort_values(by="value")
df2.head(15)

gate_incorrect= df2["value"] != "."
df2=df2[gate_incorrect]
df2.head(5)
df2["value"]=pd.to_numeric(df2["value"])
df2.info()
df2=df2.sort_values(by="date")
df2["year_diff"]=df2["date"].apply(lambda x : str(x.year) + '-' + str(x.year-1) )
df2["value_diff"]=df2["value"].diff()
df2.tail(10)

temp=df2.sort_values(by="date")
plt.plot(temp["date"],temp["value_diff"])
plt.xticks(rotation=60)

#temp=df2[(df2["date"]>='1940') & (df2["date"]>='1950')].sort_values(by="date",ascending=False)
temp=df2[(df2["date"]>='1940') & (df2["date"]<='1950')]
plt.plot(temp["year_diff"],temp["value_diff"])
plt.xticks(rotation=45)
temp=df2[(df2["date"]>='1976') & (df2["date"]<='1986')]
plt.plot(temp["year_diff"],temp["value_diff"])
plt.xticks(rotation=45)
temp=df2[(df2["date"]>='1990') & (df2["date"]<='2000')]
plt.bar(temp["year_diff"],temp["value_diff"])
plt.xticks(rotation=45)
temp=df2[(df2["date"]>='2004') & (df2["date"]<='2010')]
plt.bar(temp["year_diff"],temp["value_diff"])
plt.xticks(rotation=45)