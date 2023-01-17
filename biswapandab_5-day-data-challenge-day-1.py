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
df = pd.read_csv("../input/DigiDB_digimonlist.csv")
print(df.describe())

df1 = pd.read_csv("../input/DigiDB_movelist.csv")
print(df1.describe())

df2 = pd.read_csv("../input/DigiDB_supportlist.csv")
print(df2.describe())
#5 Day Data Challenge: Day 2
#Find the Numerical Value to plot a Histogram.
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
ds = pd.read_csv('../input/DigiDB_digimonlist.csv')
ds.describe()

import seaborn as sns
x = ds['Lv 50 HP']
a = sns.distplot(x, kde = False).set_title("Histogram")
# a.set_title("Histogram")