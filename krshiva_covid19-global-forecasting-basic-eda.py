# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-1/train.csv')
#number of rows and columns

print(df.shape)
# data columns 

df.info()
# data sample

df.head()
#Renaming the columns

df.rename(columns={'Province/State':'part_of_country','Country/Region':'country'},inplace=True)

df.tail()
# drop duplicates 

df.drop_duplicates(['Date', 'country','Lat','Long'],keep = False ,inplace=True)
df.shape
country_with_confirmed_cases = (df.groupby(['country','ConfirmedCases'], sort= True).size())
country_with_fatilities = (df.groupby(['country','Fatalities'], sort= True).size())

print(country_with_fatilities)
corr=df.corr(method='spearman')

print(corr)
# Generate a mask for the upper triangle

mask = np.triu(np.ones_like(corr, dtype=np.bool))

# Generate a custom diverging colormap

cmap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(corr, mask=mask, cmap=cmap, annot=True,vmax=.3, center=0,

            square=True, linewidths=.5, cbar_kws={"shrink": .5})