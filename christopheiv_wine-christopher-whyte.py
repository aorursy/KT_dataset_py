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
import pandas as pd
from matplotlib import pyplot as plt
%matplotlib inline
import seaborn as sn

print("environment works")
wine = pd.read_csv("../input/winemagdata-first150k/winemag-data_first150k.csv", sep=",", header=0)
wine.head(12)
sn.boxplot(data=wine.head(12))
co=wine.corr()
sn.heatmap(co,annot=True,linewidths=1.0)

sn.violinplot(x="price",y="points", data=wine.head(12))
co=wine.corr()
sn.heatmap(co,annot=True,linewidths=1.0)

#Points and price is direct correlation. you can infer that the higher points, the higher price is. 