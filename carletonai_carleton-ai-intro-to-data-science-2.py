# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.



# Get the Data Dictonary here

# https://open.canada.ca/data/en/dataset/1eb9eba7-71d1-4b30-9fb1-30cbdab7e63a



#Code for line of best fit: plt.plot(np.unique(df["C_HOUR"]), np.poly1d(np.polyfit(df["C_HOUR"], df["C_VEHS"], 1))(np.unique(df["C_HOUR"])), color="red")
df = pd.read_csv("/kaggle/input/ncdb-2014/NCDB_2014.csv", na_values=["UUUU", "UU", "U", "XXXX", "XX", "X", "QQQQ", "QQ", "Q"])
df.head()
df = df.dropna()

df = df.reset_index(drop=True)

df = df.astype({"C_HOUR": int, "C_RALN": int, "C_RSUR": int})

df.head()
plt.boxplot(df)