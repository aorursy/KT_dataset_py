import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib as mpl # To plot data

import seaborn as sns # to plot data + some regression

import statsmodels.api as sm # Ordinary Least Squares (OLS) Regression and some other

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
names = ['id', 'title', 'year', 'rating', 'votes', 'length', 'genres']

data = pd.read_csv('/kaggle/input/imbd-top-10000/imdb_top_10000.txt', sep="\t", names=names)

data
data.head()
data.head(3)
data.tail()
data.info()
data.describe()
data.to_csv('test.csv', header=True, index=True, sep=',')
data.sort_values(by='rating')
data.sort_values(by='rating', ascending=True)
sample_data = {

    'tv': [230, 44, 17],

    'radio': [37, 39, 45],

    'news': [69, 45, 69],

    'sales': [22, 10, 9]

}
data2 = pd.DataFrame(sample_data)
data2
del data2
data2

# this code play some error due to it can't find variable data2 because i has delete to show you "del data2" works.
data['title']
data[['title', 'year']]
data['rating'].mean()
data['rating'].max()
data['rating'].min()
data['genres'].unique()
data['rating'].value_counts()
data['rating'].value_counts().sort_index()
data['rating'].value_counts().sort_index(ascending=False)
# We can't write code "import matplotlib as mpl" because i had written above.
data.plot()
data.plot(kind='scatter', x='rating', y='votes')


data.plot(kind='scatter', x='rating', y='votes', alpha=0.3)
data['rating'].plot(kind='hist')
# We can't write code "import seaborn as sns" because i had written above.
sns.lmplot(x='rating', y='votes', data=data)
sns.pairplot(data)
# We can't write code "import statsmodels.api as sm" because i had written above.
results = sm.OLS(data['votes'], data['rating']).fit()
results.summary()
data[data['year'] > 1995]
data['year'] > 1995
data[data['year'] == 1966]
data[(data['year'] > 1995) & (data['year'] < 2000)]
data[(data['year'] > 1995) | (data['year'] < 2000)]
data[(data['year'] > 1995) & (data['year'] < 2000)].sort_values(by='rating', ascending=False).head(10)
data.groupby(data['year'])['rating'].mean()
data.groupby(data['year'])['rating'].max()
data.groupby(data['year'])['rating'].min()
# Answer of Question 1

data[data['year'] == 1996].sort_values(by='rating', ascending=False).head()
# Answer of Question 2

data[data['rating'] == data['rating'].max()]
# Answer of Question 3

data.sort_values(by='votes', ascending=False).head()
# Answer of Question 4

data[(data['year'] >= 1960) & (data['year'] < 1970)].groupby(data['year'])['rating'].mean()
data['formatted title'] = data['title'].str[:-7]
data.head()
data['formatted title'] = data['title'].str.split(' \(').str[0]
data.head()
data['formatted length'] = data['length'].str.replace(' mins.', '').astype('int')
data.head()
sns.pairplot(data)
data[data['formatted length'] == 0]