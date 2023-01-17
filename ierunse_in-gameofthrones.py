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
got = pd.read_csv("../input/character-deaths.csv", sep=",", header=0)
print(got.shape)
# Any results you write to the current directory are saved as output.
print(got.describe())
got.plot(kind="box", subplots = True, layout=(4,5), sharex=False, sharey = False)
got.hist()
ml.show()
cor = got.corr()
sn.heatmap(cor, annot = True, linewidths=1.0)
sn.violinplot(x="Gender", y="Death Chapter", data=got, size=10)