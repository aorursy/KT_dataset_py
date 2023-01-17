# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from scipy.stats import ttest_ind

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
cereals = pd.read_csv("../input/cereal.csv")
cereals.describe()
cereals.columns
cereals.head()
# to perform t-test, ensure the column is normally distributed. validation creteria: most of the  points should be on center diagonal.
from scipy.stats import probplot # for a qqplot
import pylab #
probplot(cereals["sugars"], dist="norm", plot=pylab)
probplot(cereals["sodium"], dist="norm", plot=pylab)
hotCereals = cereals["sodium"][cereals["type"]=="H"]
coldCereals = cereals["sodium"][cereals["type"]=="C"]
ttest_ind(hotCereals, coldCereals, equal_var=False)
print("Mean sodium for the hot cereals:")
print(hotCereals.mean())

print("Mean sodium for the cold cereals:")
print(coldCereals.mean())
import matplotlib.pyplot as plt
plt.hist(coldCereals,label='cold')
plt.hist(hotCereals,label='hot')
plt.title("sodium content in mg")
plt.legend(loc="upper right")
