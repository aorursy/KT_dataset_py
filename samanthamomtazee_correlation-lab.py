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

world_happiness_report = pd.read_csv("../input/world-happiness-report-2019/world-happiness-report-2019.csv")
world_happiness_report.describe()
world_happiness_report.head(10)
world_happiness_report[['Positive affect', 'Freedom', "Generosity"]].corr()
positive = []



for x in world_happiness_report["Positive affect"].iteritems():

    temp = x[1]

    if  np.isnan(temp):

        continue

    else:

        positive.append(temp)
freedom = []



for x in world_happiness_report["Freedom"].iteritems():

    temp = x[1]

    if  np.isnan(temp):

        continue

    else:

        freedom.append(temp)
from scipy import stats

pearson_coef, p_value = stats.pearsonr(freedom, positive)



# Pearson coefficient / correlation coefficient - how much are the two columns correlated?

print(pearson_coef)



# P-value - how sure are we about this correlation?

print(p_value)
# library & dataset

import seaborn as sns

import matplotlib.pyplot as plt

 

# use the function regplot to make a scatterplot

sns.regplot(x= freedom, y=positive,marker="*")

plt.show()