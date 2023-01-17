#importing Pandas. 


import pandas as pd
# loading the dataset

kaggle = pd.read_csv(r'../input/kaggle_datasets.csv')
#Displaying the first few lines of Dataset
kaggle.head()
# Discovering the type of the dataset. In this case, the pandas recognize it as a DataFrame. It's basically a table!

type(kaggle)
#The Shape property shows the size of the dataset. This dataset has 8036 rows and 14 columns

kaggle.shape
# Showing the columns
kaggle.columns

kaggle.describe(include='all')
# Here we use the count () method, allows us to see if all the columns are completely filled

kaggle.count()

pd.value_counts(kaggle['views'])
#There are only 20 datasets with 0 views on the Kaggle website

pd.value_counts(kaggle['views'] == 0)
#Using .loc (), we can know what datasets are 0 views
kaggle.loc[kaggle['views']==0]
#Here we are seeing datasets with more than 100 views and less than 100 downloads
kaggle.loc[(kaggle['views'] >= 100) & (kaggle['downloads'] <= 100)]
#Dataset with highest number of views
kaggle.loc[kaggle['views'] == 429745]
# the 10 datasets with the highest number of views


kaggle.sort_values(['views'], ascending = False).head(10)
# the 10 datasets with the highest number of downloads (which are the ones with the largest number of views)

kaggle.sort_values(['downloads'], ascending = False).head(10)
kaggle[kaggle['kernels']== 1].count()
kaggle.nunique()
#This chart shows the distribution of datasets considering views and downloads. It is clear that most datasets
# has less than 10k downloads
# Only 4 datasets have more than 300k views
kaggle.plot(figsize = (10,10), kind = 'scatter', x = 'views', y = 'downloads')
#These are from Datasets with more than 300k views
kaggle.loc[kaggle['views'] > 300000]
#Here we plot the obvious relationship between number of views and downloads. Using a heatmap. This explains why the same datasets appear in top views and downloads.

import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize = (10,10))
sns.heatmap(kaggle.corr(), annot = True, fmt = '.2f', cbar = True)
plt.xticks(rotation = 90)
plt.yticks(rotation = 0)
# What are the biggest contributors?
top_owner = pd.value_counts(kaggle['owner']).head(10)
top_owner
type(top_owner)
top_owner.plot(figsize = (12,8), kind = 'bar', title = 'Top Contributors')
# Out of curiosity, a little analysis of the datasets of the user Jacob Boysen. He posted some very interesting sets.

kaggle.loc[kaggle['owner'] == 'Jacob Boysen']
#To facilitate my analysis, I created a new variable with Jacob's data


jacob = kaggle.loc[kaggle['owner'] == 'Jacob Boysen']
#Seeing the 10 datasets with more views than Jacob posted

jacob.sort_values(['views'], ascending = False).head(10)
jacob.count()
# Here we see the 29 Jacob datasets sorted by number of views

top10 = jacob[['title', 'views']].set_index('title').sort_values('views', ascending=True)
top10.plot(kind = 'barh',figsize = (15,15), grid = True, color = 'blue', legend = True)
