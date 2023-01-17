# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sn #data visualization
sn.set(color_codes = True, style="white")
import matplotlib.pyplot as ml #data visualization
import warnings
warnings.filterwarnings("ignore")
poke = pd.read_csv("../input/got18/character-deaths.csv", sep=",", header=0)
print(poke.shape)
print(poke.head(10))
print(poke.corr())
co= poke.corr()
sn.heatmap(co)