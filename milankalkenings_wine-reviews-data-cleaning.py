import numpy as np

import pandas as pd

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

import matplotlib.colors

import seaborn as sns

sns.set_style("whitegrid", {'axes.grid' : False})

from sklearn.neighbors import KNeighborsRegressor



df = pd.read_csv('../input/wine-reviews/winemag-data-130k-v2.csv')

df.head(3)
# NOTE: somehow, the heatmap of my dataset doesn't show the null value sin the columns 

# 'country', 'province' and 'variety'. Please help me to fix this.

def nullscan(df_check, save=False):

    '''

    df: a dataframe on which we want to perofrm the nullscan

    save: determines, whether you want to save the .png of the plot or not

    

    plots the rate of null values per column in a dataframe using 

    a seaborn heatmap and a barplot.

    '''    

    # a df with the same size of the original dataframe, containing True in cells containing NUll values.

    # and False in all the other cells.

    df_nulls = df_check.isna()

    # a series containing the sum of all values within a column having the column names as indices.

    # True is interpreted as 1 and False is interpreted as 0 

    nulls_per_col = df_nulls.sum(axis=0)

    # the rate makes it way more interpretable:

    nulls_per_col /= len(df_check.index)



    with plt.style.context('dark_background'):

        fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(8, 10))

    

        # ax1 is losely based on: https://www.kaggle.com/ipshitagh/wine-dataset-data-cleaning

        # NOTE: I could have used the cmap viridis or anything else instead, 

        # but I want to make clear that you can use any customized cmap as well.

        vir = matplotlib.cm.get_cmap('viridis')

        colormap = matplotlib.colors.ListedColormap([vir(0), 'orange'])

        sns.heatmap(df_check.isnull(), cmap=colormap, cbar=False, yticklabels=False, ax=ax1)

    

        nulls_per_col.plot(kind='bar', color='orange', x=nulls_per_col.values, 

                           y=nulls_per_col.index, ax=ax2, width=1, linewidth=1, 

                           edgecolor='black', align='edge', label='Null value rate')

        

        ax2.set_ylim((0,1))

        # centered labels

        labels=df_check.columns

        ticks = np.arange(0.5, len(labels))

        ax2.xaxis.set(ticks=ticks, ticklabels=labels)

    

        # hide spines:

        # NOTE: I could have used ax2.set_frameon(False), 

        # but I wanted the bottom and the left spine to stay white.

        ax2.spines['top'].set_color('black')

        ax2.spines['right'].set_color('black')

        

        

        

        # workaround to visualize very small amounts of null values per col

        na_ticks = ticks[(nulls_per_col > 0) & (nulls_per_col < 0.05)]

        if (len(na_ticks) > 0):

            ax2.plot(na_ticks, [0,]*len(na_ticks), 's', c='orange', markersize=10, 

                     label='Very few missing values')

    

        fig.suptitle('Null Value Rate per Column', fontsize=30, y=1.05)

        ax2.legend()

        fig.tight_layout() 

        if(save):

            plt.savefig('nullscan.png')

        plt.show()

nullscan(df)

# drop all rows with Null values in 'country', 'province' OR 'variety':

df = df.dropna(subset=['country', 'province', 'variety'])



# one could even drop all rows containing 'too many' Null values:

# nrows_before = len(df.index)

# na_allowed = int(len(df.columns)/3)

# thresh = int(len(df.columns)) - na_allowed

# df = df.dropna(axis=0, thresh=thresh)

# nrows_afterwards = len(df.index)

nullscan(df, save=True)
df = df.drop('region_2', axis=1)



nullscan(df)
designation = df['designation'].value_counts().head(20) / len(df.index)



fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8,10), sharey=True)

ax.barh(y=designation.index, width=designation.values, color='orange')

ax.set_title('Occurance of Designations in the Dataset', fontsize=20)

ax.set_xlabel('Occurance in the Dataset')

ax.set_ylabel('Designation')

ax.set_facecolor('black')
df[['designation', 'region_1']] = df[['designation', 'region_1']].fillna('Unknown')

nullscan(df)
only_name = df.loc[df['taster_twitter_handle'].isnull() & df['taster_name'].notna(), 

                   ['taster_name', 'taster_twitter_handle']]

num_only_name = len(only_name.index)



only_twitter = df.loc[df['taster_name'].isnull() & df['taster_twitter_handle'].notna(), 

                      ['taster_name', 'taster_twitter_handle']]

num_only_twitter = len(only_twitter.index)



print(f'rows containing a name but no twitter handle: {num_only_name}'

      + f'\nrows containing a twitter handle but no taster name: {num_only_twitter}')
twitter_per_name = df.groupby('taster_name')['taster_twitter_handle'].nunique()



labels = twitter_per_name.index

#sizes = twitter_per_name



fig, ax = plt.subplots(figsize=(8,10))



twitter_per_name.plot(kind='barh', ax=ax, color='orange')

ax.set_xticks([0,1])

ax.set_xlabel('Number of Twitter Handles')

ax.set_ylabel('Taster')

ax.set_title('Twitter handles per taster', fontsize=20)

ax.set_facecolor('black')
df = df.drop('taster_twitter_handle', axis=1)

df['taster_name'] = df['taster_name'].fillna('Unknown')

nullscan(df)
df_cleanup = df.loc[:, ['price', 'points', 'country', 'taster_name']]

encoded = pd.get_dummies(df_cleanup[['country', 'taster_name']], prefix=['country', 'taster_name'])

df_cleanup = pd.concat([df_cleanup.drop(['country', 'taster_name'], axis=1), encoded], axis=1)



# training data

df_cleanup_known = df_cleanup.loc[df_cleanup['price'].notnull(), :]

X_known = df_cleanup_known.drop('price', axis=1)

y_known = df_cleanup_known['price']



# prediction data

df_cleanup_unknown = df_cleanup.loc[df_cleanup['price'].isnull(), :]

X_unknown = df_cleanup_unknown.drop('price', axis=1)
knn_cleanup = KNeighborsRegressor(n_neighbors=10)

knn_cleanup.fit(X=X_known, y=y_known)
df_known = df.loc[df['price'].notnull(),:]

df_predicted = df.loc[df['price'].isnull(),:]

# to evade SettingWithCopyWarning

df_predicted = df_predicted.drop('price', axis=1)

df_predicted['price'] = knn_cleanup.predict(X_unknown)

df = pd.concat([df_known, df_predicted], axis=0, ignore_index=True)



# shuflle the dataset along rows

df = df.sample(frac=1).reset_index(drop=True)







nullscan(df)

df.head()
fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(8, 10))

df.boxplot(column='price', ax=ax1)

ax.set_title('Outliers in Price?')