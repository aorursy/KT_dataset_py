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

data = pd.read_csv("../input/college-data/data.csv")
data.head()
data[["apps", "accept", "enroll", "top10perc", "top25perc", "outstate", "phd", "grad_rate", "room_board"]].corr()
from scipy import stats

pearson_coef, p_value = stats.pearsonr(data["outstate"], data["room_board"])



print(pearson_coef)



print(p_value)
import seaborn as sns

import matplotlib.pyplot as plt



sns.regplot(x=data["outstate"], y=data["room_board"], marker= "P", color = "peachpuff", line_kws={'color':'chocolate'})

plt.show()

# library & dataset

import seaborn as sns

import matplotlib.pyplot as plt

#df = sns.load_dataset('data')

 

# Make boxplot for one group only

sns.boxplot(x=data["outstate"], color="mistyrose", linewidth = 5)

plt.show()

# library & dataset

import seaborn as sns

import matplotlib.pyplot as plt

#df = sns.load_dataset('data')

 

# Basic 2D density plot

sns.set_style("white")

#sns.plt.show()

 

# Custom it with the same argument as 1D density plot

#sns.kdeplot(data, cmap="Reds", shade=True, bw=.15)

 

# Some features are characteristic of 2D: color palette and wether or not color the lowest range

sns.kdeplot(data["outstate"], data["room_board"], cmap="Blues", shade=True, shade_lowest=True)

plt.show()
