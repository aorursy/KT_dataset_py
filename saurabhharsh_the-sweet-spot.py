# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
#print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/flavors_of_cacao.csv")
df.head()
df['Cocoa\nPercent'] = df['Cocoa\nPercent'].apply(lambda x : float(x[:-1]))
columnNames = df.columns.values
countryCol = columnNames[5]
originCol = columnNames[8]
df.replace(to_replace = {countryCol : {'Amsterdam' : 'Netherlands',
                                        'Eucador' : 'Ecuador'},
                         originCol : {'Domincan Republic' : 'Dominican Republic'}}, inplace = True)
import seaborn as sns
fig, ax = sns.mpl.pyplot.subplots(figsize=(15,6))
sns.distplot(df['Rating'])
cocoaPct = columnNames[4]
fig, ax = sns.mpl.pyplot.subplots(figsize=(15,6)) #fig, ax = sns.mpl.pyplot.subplots()
sns.swarmplot(ax = ax, y = cocoaPct, x = 'Rating', data = df, orient = 'v')
dfGrouped = df.groupby(countryCol)
nReviews = dfGrouped.size()
fig, ax = sns.mpl.pyplot.subplots(figsize=(15,15))
sns.countplot(ax = ax, y = countryCol, data = df, orient = 'h', order = nReviews.sort_values(ascending = False).index.values)
meanRating = dfGrouped['Rating'].mean()
meanRating = meanRating[nReviews > 10].sort_values(ascending = False)

fig, ax = sns.mpl.pyplot.subplots(figsize=(15,10))
sns.boxplot(ax = ax, x =  'Rating', y = columnNames[5], data = df, order = meanRating.index.values, color = 'yellow')

originCol = columnNames[8]
nReviewsByOrigin = df.groupby(originCol).size().sort_values(ascending = False)
fig, ax = sns.mpl.pyplot.subplots(figsize=(15,25))
sns.countplot(ax = ax, y = originCol, data = df, orient = 'h', order = nReviewsByOrigin.index.values)
meanByOrigin = df.groupby(originCol)['Rating'].mean()[nReviewsByOrigin > 10].sort_values(ascending = False)
fig, ax = sns.mpl.pyplot.subplots(figsize=(15,10))
sns.boxplot(ax = ax, x =  'Rating', y = originCol, data = df, order = meanByOrigin.index.values, color = 'yellow')