# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/globalterrorismdb_0617dist.csv', encoding = "ISO-8859-1")

df.info()
col_names = df.columns

print(col_names)
d_2016 = df[df["iyear"] == 2016]

d_1970 = df[df["iyear"] == 1970]

d_1970_us = d_1970[d_1970["country_txt"] == "United States"]

d_2016_us = d_2016[d_2016["country_txt"] == "United States"]



d_1990 = df[df["iyear"] == 1990]

d_1990_col = df[df["country_txt"] == "Colombia"]



d_2016.shape
d_2016_us["nkill"].sum()
print(d_2016["country_txt"].value_counts())

print(d_1970["country_txt"].value_counts())
print(d_1970["attacktype1_txt"].value_counts())

print(d_2016["attacktype1_txt"].value_counts())
print(d_1970_us["attacktype1_txt"].value_counts())

print(f"Number Killed: {d_1970_us['nkill'].sum()}")



print(d_2016_us["attacktype1_txt"].value_counts())

print(f"Number Killed: {d_2016_us['nkill'].sum()}")
d_1970_us["attacktype1_txt"].value_counts().plot(kind='pie')

plt.show()