# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

auto_mpg = pd.read_csv("../input/autompg-dataset/auto-mpg.csv")

print(auto_mpg)
auto_mpg.describe()
auto_mpg.info()
import matplotlib.pyplot as plt

x=auto_mpg["cylinders"]

plt.hist(x,bins=10)

plt.show()

y=auto_mpg[["cylinders","horsepower"]]

y.boxplot()

plt.show()


x=auto_mpg["cylinders"]

y=auto_mpg["horsepower"]



plt.scatter(x,y,alpha=0.5)

plt.show()

             

             
import numpy as np

from scipy import stats 

x=auto_mpg["cylinders"]

y=auto_mpg["horsepower"]



print("mean",np.mean(x))

print("meadian",np.median(x))

print("mode",stats.mode(x))
auto_mpg.columns
auto_mpg.head()