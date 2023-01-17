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
country_list = ["Spain","France"]
population_list = ["11","12"]
label_list = ["country","population"]
column_list = [country_list,population_list]
zipped = list(zip(label_list,column_list))
dic = dict(zipped)
df = pd.DataFrame(dic)
df
df["capital"] = ["Madrid","Paris"]
df
df["income"] = 0
df
data1 = data.loc[:,["Attack","Defense","Speed"]]
data1.plot()
plt.show()
data1.plot(subplots=True)
plt.show()
data1.plot(kind="hist",bins=50 , range=(0,250), normed=True)
plt.show()
df["num"]=[12.3,111.5]
date_list = ["1996-02-15","2004-10-03"]
date_obj = pd.to_datetime(date_list)
df = df.set_index(date_obj)
df.resample("A").mean()