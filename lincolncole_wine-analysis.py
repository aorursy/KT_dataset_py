# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sn # data visualization
%matplotlib inline
sn.set (color_codes = True, style="white")
import matplotlib.pyplot as ml #data visualization
import warnings
warnings.filterwarnings("ignore")

print ("Environment works!!!")
# Let's import our data 
wine = pd.read_csv("../input/winemag-data-130k-v2.csv", sep = ",", header = 0)
# Let's take a look at the shape and size of our data

wine.shape
# Let's take a look at this data

wine.head(10)
# Let's plot a histogram

wine.hist()
ml.show()
# Let's plot a box plot

sn.boxplot(data=wine)
# Let's look at a violin plot comparing countries with points

ml.subplots(figsize=(40,40))
sn.violinplot(x="country", y="points", data=wine)
# Lets have a look at the heatmap

ml.subplots(figsize=(10,10))
co = wine.corr()
sn.heatmap(co, annot=True, linewidths=1.0)