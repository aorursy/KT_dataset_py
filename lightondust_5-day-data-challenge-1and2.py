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
data = pd.read_csv("../input/FAO.csv", encoding='latin1') 
data.shape
data.columns
data.head()
data.describe()
data["Item"].value_counts()
import matplotlib.pyplot as plt
data_to_plot = data["Y2004"]
data_to_plot.head()
help(plt.hist)
plt.hist(data_to_plot)
data.isnull().any(axis=0)
data_to_plot = data["Y2013"]
plt.hist(data_to_plot)
plt.hist(data_to_plot, range=[0, 100])
plt.title("Range 0-100")
plt.hist(data_to_plot, range=[10000, 20000])
data.isnull().sum()
help(data.isnull)
