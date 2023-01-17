# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sn # data visualization
sn.set(color_codes = True, style="white")
import matplotlib.pyplot as ml #data visualization
import warnings
warnings.filterwarnings("ignore")
happiness = pd.read_csv("../input/2017.csv", sep=",", header=0)
print(happiness.shape)
print(happiness.head)
happiness.corr()
print(happiness.describe())
co = happiness.corr()
ax=sn.heatmap(co)
ax.set_title("Heatmap")