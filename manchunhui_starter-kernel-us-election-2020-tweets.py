import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df1=pd.read_csv('/kaggle/input/us-election-2020-tweets/hashtag_donaldtrump.csv', lineterminator='\n')

df1.head()
df1.describe()
g=sns.pairplot(df1, vars=['tweet_id','likes','retweet_count','user_id','user_followers_count','lat','long'], 

               plot_kws = {'alpha': 0.6, 's': 30, 'edgecolor': 'black'}, height=1.5, dropna=True)

g.map_upper(sns.scatterplot, color='red')

g.map_lower(sns.scatterplot, color='red')

g.map_diag(plt.hist, color='red')
df1.isnull().sum()
df2=pd.read_csv('/kaggle/input/us-election-2020-tweets/hashtag_joebiden.csv', lineterminator='\n')

df2.head()
df2.describe()
sns.pairplot(df2, vars=['tweet_id','likes','retweet_count','user_id','user_followers_count','lat','long'], 

             plot_kws = {'alpha': 0.6, 's': 30, 'edgecolor': 'k'}, height=1.5, dropna=True)
df2.isnull().sum()