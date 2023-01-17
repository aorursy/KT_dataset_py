# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
h_r_2015 = pd.read_csv('../input/2015.csv')
h_r_2016 = pd.read_csv('../input/2015.csv')
h_r_2015.head()

h_r_2016.head()
fig, axes = plt.subplots(figsize=(10, 7))

corr = h_r_2016.corr()

mask = np.zeros_like(corr)

mask[np.triu_indices_from(mask)] = True

with sns.axes_style("white"):

    ax = sns.heatmap(corr,linewidths=1,annot=True, mask=mask, vmax=.3, square=True)

axes.set_title("2016")
sns.pairplot(h_r_2016[['Happiness Score','Economy (GDP per Capita)','Family','Health (Life Expectancy)']])
h_r_2015['Year'] = '2015'

h_r_2016['Year'] = '2016'

happiness_report_2015_2016 = pd.concat([h_r_2015[['Happiness Score','Region','Year']],h_r_2016[['Happiness Score','Region','Year']]])

happiness_report_2015_2016.head()
from scipy.stats import kendalltau
h_r_2016[['Happiness Score']].head()
df = pd.DataFrame()

df['x'] = h_r_2016[['Happiness Score']]

df['y'] = h_r_2016[['Family']]

df.head()
sns.lmplot('x', 'y', data= df, fit_reg = False)
f, ax = plt.subplots(figsize=(6, 6))

cmap = sns.cubehelix_palette(as_cmap=True, dark=0, light=1, reverse=True)

sns.kdeplot(df.x, df.y, cmap=cmap, n_levels=60, shade=True);
sns.kdeplot(df.y, df.x)