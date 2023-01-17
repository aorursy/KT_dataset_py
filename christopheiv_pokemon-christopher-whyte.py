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

print ("environment works")
pokomon = pd.read_csv("../input/Pokemon_cw.csv", sep = ",", header=0)
pokomon.head (15)
sn.boxplot(data=pokomon.head(15))
co = pokomon. corr()
sn.heatmap(co, annot=True, linewidths=1.0)
sn.violinplot (x="Speed",y="Attack", data=pokomon.head(15))
#co = pokomon. corr()
sn.heatmap(co, annot=True, linewidths=1.0)
#You can infer the Sp.Defense and the Sp. Attack has direct correlation (.51)