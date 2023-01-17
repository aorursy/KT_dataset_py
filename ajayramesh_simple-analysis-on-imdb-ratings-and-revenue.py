import warnings

warnings.filterwarnings("ignore")



# Utils

from os import path



# Data 

import numpy as np

import pandas as pd



# Viz

import matplotlib as mpl

import matplotlib.pyplot as plt

import seaborn as sns



%matplotlib inline

sns.set(style="darkgrid")
data_path = ""

df = pd.read_csv(data_path+"../input/imdb-data/IMDB-Movie-Data.csv")

df.head(3)
df.columns
df.columns = ['rank', 'title', 'genre', 'description', 'director', 'actors', 'year',

       'runtime_min', 'rating', 'votes', 'revenue_in_millions',

       'metascore']
df.head()
df.shape
df.info()
pd.DataFrame(df.isnull().sum(), columns=["Columns"])
df.dropna(inplace=True)
def ecdf(data):

    n = len(data)

    x = np.sort(data)

    y = np.arange(1, n+1) / n

    return x, y
fig, axes = plt.subplots(3, 2, figsize=(14, 14))



# PLOT 1 

axes[0, 0].set_title('How many movies earned more the $500M ? ')

_ = axes[0, 0].hist(df.revenue_in_millions, bins=np.arange(0, 800, 50))

axes[0, 0].set(xlabel='Revenue in millions', ylabel='Number of Movies/Shows')



# PLOT 2 

axes[0, 1].set_title('How many movies rated above 7.5 ?')

_ = axes[0, 1].hist(df.rating, bins=np.arange(0, 10, .5))

_ = axes[0, 1].set( xlabel='IMDB Rating')



# PLOT 3 

axes[1, 0].set_title('How many movies released every year and rated above 7.5 ?')

chart = sns.swarmplot(y='rating', x='year', data=df, ax=axes[1,0])

_ = axes[1, 0].set( ylabel='IMDB Rating', xlabel='Year Released')

chart.set_xticklabels(chart.get_xticklabels(), rotation=90)





# PLOT 4 

axes[1, 1].set_title('ECDF Analysis on IMDB Rating ')

x_vers, y_vers = ecdf(df.rating)

_ = axes[1, 1].plot(x_vers, y_vers, marker='.', linestyle='none')

_ = axes[1, 1].set(xlabel='IMDB Rating', ylabel='ECDF')



# PLOT 5 

axes[2, 0].set_title('Year Vs Ratings ')

chart = sns.boxplot(x='year', y='rating', data=df,  ax=axes[2,0])

chart.set_xticklabels(chart.get_xticklabels(), rotation=90)

_ = axes[2, 0].set( ylabel='IMDB Rating', xlabel='Year Released')



# PLOT 6

axes[2, 1].set_title('Revenue Vs Rating ')

_ = axes[2, 1].plot(df.rating, df.revenue_in_millions, marker='.', linestyle='none')

_ = axes[2, 1].set( xlabel='IMDB Rating', ylabel='Revenue in millions')





plt.tight_layout(pad=2.0)

plt.show()
df_metrics = pd.DataFrame({

    "Rating":[np.mean(df.rating), np.median(df.rating)],

    "Revenue":[np.mean(df.revenue_in_millions), np.median(df.revenue_in_millions)]

   }, index=["Mean", "Median"])
df_metrics
def pearson_r(x, y):

    """Compute Pearson correlation coefficient between two arrays."""

    # Compute correlation matrix: corr_mat

    corr_mat = np.corrcoef(x, y)



    # Return entry [0,1]

    return corr_mat[0,1]



# Compute Pearson correlation coefficient for I. versicolor

r = pearson_r(df.rating, df.revenue_in_millions)



# Print the result

print(r)