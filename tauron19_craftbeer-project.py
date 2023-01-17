# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sn # data visualisation
sn.set(color_codes = True, style = "white")
import matplotlib.pyplot as ml # data visualisation
import warnings
warnings.filterwarnings("ignore")
beer = pd.read_csv("../input/beers.csv", sep=",", header = 0)
print(beer.shape)

# Any results you write to the current directory are saved as output.
beer.head(10)
beer .corr() 
sn.lmplot(x='ounces', y='abv', data=beer)
sn.distplot(beer.ounces)
co = beer.corr()
sn.heatmap(co)
sn.jointplot(x='abv', y='ounces', data=beer)
sn.countplot(x='style', data=beer)
sn.violinplot(x="ounces", y="abv", data= beer, size=10)