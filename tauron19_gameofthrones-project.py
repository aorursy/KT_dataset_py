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
import os
dt = pd.read_csv("../input/character-deaths.csv", sep=",", header = 0)
print(dt.shape)


# Any results you write to the current directory are saved as output.
dt.head(10)
dt .corr()
dt.columns = [col.replace(' ', '_').lower() for col in dt.columns]
print(dt.columns)
dt.fillna(method='bfill', inplace=True)
co = dt.corr()
sn.heatmap(co)
sn.distplot(dt.book_of_death)
sn.countplot(data = dt, y= "gender");
sn.distplot(dt.death_year)
sn.countplot(data = dt, x= "nobility");