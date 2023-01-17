# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df = df = pd.read_csv(r'../input/drinks.csv')
df.head()
df.groupby('continent').beer_servings.mean().sort_values().plot.bar()
df[df.continent == 'Africa'].beer_servings.mean()
df.groupby('continent').beer_servings.max().sort_values()
df.groupby('continent').beer_servings.count().sort_values().plot.bar()
df.groupby('continent').beer_servings.agg(['count', 'min', 'max', 'mean'])
df.groupby('continent').mean()
df.groupby('continent').mean().plot(kind = 'bar')
df[(df.wine_servings >= 20) & (df.continent == 'Asia')]
df[(df.wine_servings >= 50) | (df.continent == 'Asia')]
