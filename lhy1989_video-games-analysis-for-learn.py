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
import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
df = pd.read_csv("../input/vgsales.csv")

df.head()
df.info()
platGenre = pd.crosstab(df.Platform,df.Genre)

platGenreTotal = platGenre.sum(axis=1).sort_values(ascending = False)

plt.figure(figsize=(8,6))

sns.barplot(y = platGenreTotal.index, x = platGenreTotal.values, orient='h')

plt.ylabel = "Platform"

plt.xlabel = "The amount of games"

plt.show()
platGenre['Total'] = platGenre.sum(axis=1)

popPlatform = platGenre[platGenre['Total']>1000].sort_values(by='Total', ascending = False)

neededdata = popPlatform.loc[:,:'Strategy']

maxi = neededdata.values.max()

mini = neededdata.values.min()

popPlatformfinal = popPlatform.append(pd.DataFrame(popPlatform.sum(), columns=['total']).T, ignore_index=False)

sns.set(font_scale=0.7)

plt.figure(figsize=(10,5))

sns.heatmap(popPlatformfinal, vmin = mini, vmax = maxi, annot=True, fmt="d")

plt.xticks(rotation = 90)

plt.show()
GenreGroup = df.groupby(['Genre']).sum().loc[:, 'NA_Sales':'Global_Sales']

GenreGroup['NA_Sales%'] = GenreGroup['NA_Sales']/GenreGroup['Global_Sales']

GenreGroup['EU_Sales%'] = GenreGroup['EU_Sales']/GenreGroup['Global_Sales']

GenreGroup['JP_Sales%'] = GenreGroup['JP_Sales']/GenreGroup['Global_Sales']

GenreGroup['Other_Sales%'] = GenreGroup['Other_Sales']/GenreGroup['Global_Sales']

plt.figure(figsize=(8, 10))

sns.set(font_scale=0.7)

plt.subplot(211)

sns.heatmap(GenreGroup.loc[:, 'NA_Sales':'Other_Sales'], annot=True, fmt = '.1f')

plt.title("Comparation of each area in each Genre")

plt.subplot(212)

sns.heatmap(GenreGroup.loc[:,'NA_Sales%':'Other_Sales%'], vmax =1, vmin=0, annot=True, fmt = '.2%')

plt.title("Comparation of each area in each Genre(Pencentage)")

plt.show()