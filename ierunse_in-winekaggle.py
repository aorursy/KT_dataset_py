# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sn # data visualization
import matplotlib.pyplot as ml #data visualization

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
wine = pd.read_csv("../input/winemag-data_first150k.csv", sep=",", header=0)
#print(wine.head)
print(wine.shape)
# Any results you write to the current directory are saved as output.
print(wine.describe())
wine.plot(kind="box", subplots = True, layout=(5,6), sharex=False, sharey = False)
sn.violinplot(x="points", y="price", data=wine, size=10)
cor = wine.corr()
sn.heatmap(cor, annot = True, linewidths=1.0)
