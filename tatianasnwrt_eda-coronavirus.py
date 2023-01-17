# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

import matplotlib.pylab as pylab

%matplotlib inline
pylab.rcParams["figure.figsize"] = (14,8)
data = pd.read_csv("/kaggle/input/coronavirusdataset/patient.csv", index_col="id")
data.head()
data.info()
sns.countplot(x='sex', data=data)

plt.xticks(rotation=90)

plt.ylabel('Count')

plt.show()
data.state.value_counts()
deceased = data[data.state == "deceased"]
sns.countplot(x='birth_year', data=deceased)

plt.xticks(rotation=90)

plt.ylabel('Count')

plt.show()
sns.countplot(x='sex', data=deceased)

plt.xticks(rotation=90)

plt.ylabel('Count')

plt.show()
infected = data[data.state != "deceased"]
infected.head()
sns.countplot(x='birth_year', data=infected)

plt.xticks(rotation=90)

plt.ylabel('Count')

plt.show()
by_country_by_year = data.groupby(["country","birth_year"]).size().unstack()

g = sns.heatmap(by_country_by_year, 

    #square=True, # make cells square

    cbar_kws={'fraction' : 0.02}, # shrink colour bar

    cmap='OrRd', # use orange/red colour map

    linewidth=1 # space between cells 

               )
infection_reason = data.groupby(["infection_reason","birth_year"]).size().unstack()

g = sns.heatmap(infection_reason, 

    #square=True, # make cells square

    cbar_kws={'fraction' : 0.02}, # shrink colour bar

    cmap='OrRd', # use orange/red colour map

    linewidth=1 # space between cells 

               )
region = data.groupby(["region","birth_year"]).size().unstack()

g = sns.heatmap(region, 

    #square=True, # make cells square

    cbar_kws={'fraction' : 0.02}, # shrink colour bar

    cmap='OrRd', # use orange/red colour map

    linewidth=1 # space between cells 

               )