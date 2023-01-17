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
battles = pd.read_csv("../input/battles.csv", sep=",",header=0)
battles.head(10)
sn.boxplot(data=battles.head(10))
sn.violinplot(x="major_death",y="major_capture",data=battles.head(10))
co = battles.corr()
sn.heatmap(co,annot=True, linewidths=1.0)