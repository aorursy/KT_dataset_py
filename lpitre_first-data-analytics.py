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
iris = pd.read_csv("../input/Iris.csv", sep=",", header=0)
print(iris.shape)

print(iris.head)#head gives a preview
print(iris.describe())
iris.hist()
ml.show()
iris.plot(kind="box", subplots = True, layout=(2,3), sharex=False, sharey = False)

sn.violinplot(x="Species", y="PetalLengthCm", data=iris, size=10)