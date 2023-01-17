# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sn#for visuals

sn.set(style="white", color_codes=True)#customizes the graphs

import matplotlib.pyplot as mp#for visuals

%matplotlib inline#how the graphs are printed out

import warnings#suppress certain warnings from libraries

warnings.filterwarnings("ignore")



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.

iris = pd.read_csv("../input/Iris.csv", sep=",", header=0)

print(iris.shape)
print(iris.head)
print(iris.describe())
iris.hist()

mp.show()
iris.plot(kind="box", subplots = True, layout=(2,3), sharex=False, sharey = False)
sn.violinplot(x="Species", y="PetalLengthCm", data=iris, size=10)
sn.FacetGrid(iris, hue="Species", size=5).map(mp.scatter, "SepalLengthCm", "SepalWidthCm").add_legend()