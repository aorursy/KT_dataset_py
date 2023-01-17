# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#1

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
pd.set_option("display.max_rows", 5)
#2

#groupby() created a group of reviews which allotted the same point values to
# the given wines. Then, for each of these groups, we grabbed the points() column 
#and counted how many times it appeared.
reviews.groupby('points').points.count()
#3

reviews.groupby('points').price.min()
#4

reviews.groupby('winery').apply(lambda df: df.title.iloc[0])
#5

# group by more than one column. For an example, here's how we would pick out the best wine by country and province
reviews.groupby(['country', 'province']).apply(lambda df: df.loc[df.points.idxmax()])
#6

# run a bunch of different functions on your DataFrame simultaneously.
reviews.groupby(['country']).price.agg([len, min, max])
#7

# Multi-index
countries_reviewed = reviews.groupby(['country', 'province']).description.agg([len])
countries_reviewed
#8

#Sorting

countries_reviewed = countries_reviewed.reset_index()
countries_reviewed.sort_values(by='len')
#9

#Sorting Descending Order

countries_reviewed.sort_values(by='len', ascending=False)