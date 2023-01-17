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
# read data

import pandas as pd



# read names and assign bnames dataframe

bnames = pd.read_csv("../input/names.csv")

bnames.head()
#Let's find top 5 popular male and female names born 2010 and later
bnames_2010 = bnames.loc[bnames['year'] >= 2010]

bnames_2010_agg = bnames_2010.groupby(['sex','name'],as_index=False)['births'].sum()

bnames_top5 = bnames_2010_agg.sort_values(['sex','births'],ascending=[True,False]).groupby('sex').head().reset_index(drop=True)

bnames_top5
bnames2 = bnames.copy()

#compute proportion of births by year

total_births_by_year = bnames2.groupby(['year'])['births'].transform('sum')

bnames2['prop_births'] = bnames2['births'] / total_births_by_year
# to plot in notebook lines

%matplotlib inline 

import matplotlib.pyplot as plt



def plot_trends(name,sex):

    bnames2.loc[(bnames2['sex'] == sex) & (bnames['name']==name)]['births'].plot(subplots=True)

    plt.show()

    return name



plot_trends('Mary','F')
