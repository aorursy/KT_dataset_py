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

# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/Pokemon.csv")
data.head(7)
data.tail(7)
data.columns
data.describe()
data.corr()
data.shape
data1 = data.head(7)
data1
melted = pd.melt(frame = data1, id_vars = "Name", value_vars = ["Attack","Defense"])
melted
melted.pivot(index="Name", columns="variable", values="value")
data2 = data.tail(7)
data3 = pd.concat([data1,data2], axis=0, ignore_index=True)
data3
data4 = data1.Attack
data5 = data1.Defense
data6 = pd.concat([data4,data5], axis=1)
data6
print(data["Type 2"].value_counts(dropna=False))
data.Speed = data.Speed.astype("float")
data.info()
data["Type 2"].dropna(inplace=True)
data["Type 2"].value_counts(dropna=False)
assert data["Type 2"].notnull().all()
data["Type 2"].fillna("empty",inplace=True)
assert data["Type 2"].notnull().all()