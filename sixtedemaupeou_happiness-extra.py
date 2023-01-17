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
base_df = pd.read_csv('/kaggle/input/world-happiness/2019.csv')

base_df = base_df.rename(columns={"Country or region": "country"})

base_df['country'] = base_df['country'].str.strip()

print(base_df.columns)

base_df
extra_df = pd.read_csv('/kaggle/input/countries-of-the-world/countries of the world.csv', decimal=',')

extra_df = extra_df.rename(columns={"Country": "country"})

extra_df['country'] = extra_df['country'].str.strip()

print(extra_df.columns)

extra_df
full_df = pd.merge(base_df, extra_df, on='country', how='left')

full_df.dropna()

full_df
full_df = full_df[['country', 'Score', 'GDP per capita', 'Social support',

       'Healthy life expectancy', 'Freedom to make life choices', 'Generosity',

       'Perceptions of corruption', 'Population', 'Area (sq. mi.)',

       'Pop. Density (per sq. mi.)', 'Coastline (coast/area ratio)',

       'Net migration', 'Infant mortality (per 1000 births)', 'Literacy (%)', 'Phones (per 1000)', 'Arable (%)',

       'Crops (%)', 'Climate', 'Birthrate', 'Deathrate',

       'Agriculture', 'Industry', 'Service']]

# full_df

spearman_cormatrix= full_df.corr(method='spearman')

spearman_cormatrix
import matplotlib.pyplot as plt

import seaborn as sns



# fig, ax = plt.subplots(ncols=1,figsize=(48, 48))

plt.rcParams["figure.figsize"] = [20, 20]

sns.heatmap(spearman_cormatrix, vmin=-1, vmax=1, center=0, cmap="viridis", annot=True)