

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python





import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt
gdp = pd.read_csv('../input/worldGDP_growth2.csv', encoding = "ISO-8859-1")
gdp
movie = pd.read_csv('../input/movie_metadata.csv')
movie.describe()
gdp.describe()
gdp_stats= gdp.describe().T
gdp_stats.head()
movie['budget'].max()
budget = movie[movie['budget']<300000000]

budget = budget[budget['budget'].isnull() == False]
sns.distplot(budget['budget'])
budget = pd.concat([budget['title_year'],budget['budget']],axis = 1)

budget.head()
budget = budget.dropna(axis = 0)

sns.jointplot(x = 'title_year', y = 'budget',data = budget)
budget = budget[budget['title_year']>1960]

sns.jointplot(x = 'title_year', y = 'budget',data = budget)
gdp_stats = gdp_stats.reset_index()

gdp_stats = gdp_stats.dropna()

gdp_stats.head()
sns.distplot(gdp_stats['mean'])
df = gdp_stats.copy()
df['index'] = df['index'].astype(str).astype(int)

df=df.rename(columns = {'index':'title_year'})

df=df.rename(columns = {'mean':'growth_rate'})



sns.jointplot(x = 'title_year', y = 'growth_rate', data = df)
sns.jointplot(x = df['growth_rate'], y = budget['budget'])
sns.jointplot(x = df['growth_rate'], y = budget['budget'])
gross = movie[['title_year','gross']].dropna(axis = 0)

gross = gross[gross['title_year']>1960]

gross.describe()
temp = pd.concat([df['growth_rate'],gross['gross']],axis = 1)
temp = temp.dropna()

temp.head()
sns.set(style="darkgrid")

sns.lmplot(x="growth_rate", y="gross", data=temp)
temp['gross'] = temp['gross']/100000

temp.head()