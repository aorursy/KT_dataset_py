#Libraries/Dependices

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns

%reload_ext autoreload
%autoreload 2
%matplotlib inline

from fastai.structured import *
from fastai.column_data import *
np.set_printoptions(threshold=50, edgeitems=20)

import os
print(os.listdir("../input"))


PATH = "../input/Video_Games_Sales_as_at_22_Dec_2016.csv"
df_Games = pd.read_csv("../input/Video_Games_Sales_as_at_22_Dec_2016.csv", index_col=0)
df_Games.head(30)
df_Games.info()
df_Games.describe()
sns.distplot(df_Games['Global_Sales'], kde=False)
sns.distplot(df_Games['Year_of_Release'].dropna())
sns.distplot(df_Games['Critic_Score'].dropna())
plt.figure(figsize=(12,8))
sns.countplot(df_Games['Platform'].sort_index())
plt.figure(figsize=(14,8))
sns.countplot(df_Games['Genre'].sort_index())
plt.figure(figsize=(12,8))
sns.countplot(df_Games['Developer'].dropna(), order = df_Games['Developer'].value_counts().iloc[:40].index)
plt.xticks(rotation=90);

sales_genres = df_Games[['Genre', 'Global_Sales']]
sales_genres.head()