# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sn # data visualization

sn.set(color_codes = True, style="white")

import matplotlib.pyplot as ml # data visualization

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

import warnings
warnings.filterwarnings("ignore")
pokee = pd.read_csv("../input/Pokemon.csv", sep=",",header=0)
print(pokee.shape)




print(pokee.describe())
pokee[pokee.Attack > 50]

pokee['Name']
pokee.ix[:10,("Name","Attack")]

print(pokee.head)
print(pokee.corr)
f,ax = ml.subplots(figsize=(18, 18))
sn.heatmap(pokee.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
sn.countplot(x='Type 1', data=pokee, order=pokee['Type 1'].value_counts().index)
ml.xticks(rotation=100)
ml.show()