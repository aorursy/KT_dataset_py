# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import matplotlib.pyplot as plt #Plotting
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/data.csv")
data
data.info(verbose=True,  memory_usage=True, null_counts=True)
data[data.isnull().any(axis=1)]
# data[!data.isnull().any(axis=1)]
notnull = data[data.notnull().any(axis=1)]
notnull["Streams"].describe()
notnull[notnull["Streams"] < 1e6].describe()
notnull[notnull["Streams"] < 1e5].describe()
notnull[notnull["Streams"] < 1e4].describe()