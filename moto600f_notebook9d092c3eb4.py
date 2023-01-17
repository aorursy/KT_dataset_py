# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from pandas import Series, DataFrame

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import codecs

with codecs.open("../input/albumlist.csv", "r", "ASCII", "ignore") as file:

    data = pd.read_table(file, delimiter=",")

data.head()
data.describe()
data.shape
years = np.arange(1950,2015)

ax = sns.factorplot(x='Year', data=data, kind='count', order=years)

ax.set_xticklabels(step=5)

plt.xticks(rotation=45)

plt.title('Distribution of Published Year')
plt.figure(figsize=(5,10))

sns.stripplot(y='Genre',x='Year',data=data, jitter=True)

plt.title('Genre Apperence', fontsize = 15)
popular_genre = data.groupby(['Genre'],as_index=False).count().nlargest(10,'Album')

sns.barplot(x='Genre',y='Album',data=popular_genre)

plt.ylabel("Count")

plt.xticks(rotation=90)

plt.title('Top 10 Genre')
popular_artist = data.groupby(['Artist'],as_index=False).count().nlargest(10,'Album')

sns.barplot(x='Artist',y='Album',data=popular_artist)

plt.ylabel("Count")

plt.xticks(rotation=90)

plt.title('Top 10 Artist')