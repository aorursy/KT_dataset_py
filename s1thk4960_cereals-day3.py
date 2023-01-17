# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import pylab
from scipy.stats import probplot
from scipy.stats import ttest_ind


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
cereal = pd.read_csv("../input/cereal.csv")
cereal.head()
probplot(cereal["sodium"], dist="norm", plot=pylab)

# Any results you write to the current directory are saved as output.
hotCereals = cereal["sodium"][cereal["type"] == "H"]
coldCereals = cereal["sodium"][cereal["type"] == "C"]
ttest_ind(hotCereals, coldCereals, equal_var=False)

print("Mean sodium for hot cereals:")
print(hotCereals.mean())
print("Mean sodium for cold cereals:")
print(coldCereals.mean())
plt.hist(coldCereals, alpha=0.5, label='cold')
plt.hist(hotCereals, label='hot')
plt.legend(loc='upper right')
plt.title("Sodium(mg) content of cereals by type")