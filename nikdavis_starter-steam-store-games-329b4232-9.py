import itertools

import re



from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt # plotting

import numpy as np # linear algebra

import os # accessing directory structure

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
print(os.listdir('../input'))
# Distribution graphs (histogram/bar graph) of column data

def plotPerColumnDistribution(df, nGraphShown, nGraphPerRow):

    nunique = df.nunique()

    df = df[[col for col in df if nunique[col] > 1 and nunique[col] < 50]] # For displaying purposes, pick columns that have between 1 and 50 unique values

    nRow, nCol = df.shape

    columnNames = list(df)

    nGraphRow = (nCol + nGraphPerRow - 1) / nGraphPerRow

    plt.figure(num = None, figsize = (6 * nGraphPerRow, 8 * nGraphRow), dpi = 80, facecolor = 'w', edgecolor = 'k')

    for i in range(min(nCol, nGraphShown)):

        plt.subplot(nGraphRow, nGraphPerRow, i + 1)

        columnDf = df.iloc[:, i]

        if (not np.issubdtype(type(columnDf.iloc[0]), np.number)):

            valueCounts = columnDf.value_counts()

            valueCounts.plot.bar()

        else:

            columnDf.hist()

        plt.ylabel('counts')

        plt.xticks(rotation = 90)

        plt.title(f'{columnNames[i]} (column {i})')

    plt.tight_layout(pad = 1.0, w_pad = 1.0, h_pad = 1.0)

    plt.show()

# Correlation matrix

def plotCorrelationMatrix(df, graphWidth):

    filename = df.dataframeName

    df = df.dropna('columns') # drop columns with NaN

    df = df[[col for col in df if df[col].nunique() > 1]] # keep columns where there are more than 1 unique values

    if df.shape[1] < 2:

        print(f'No correlation plots shown: The number of non-NaN or constant columns ({df.shape[1]}) is less than 2')

        return

    corr = df.corr()

    plt.figure(num=None, figsize=(graphWidth, graphWidth), dpi=80, facecolor='w', edgecolor='k')

    corrMat = plt.matshow(corr, fignum = 1)

    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)

    plt.yticks(range(len(corr.columns)), corr.columns)

    plt.gca().xaxis.tick_bottom()

    plt.colorbar(corrMat)

    plt.title(f'Correlation Matrix for {filename}', fontsize=15)

    plt.show()

# Scatter and density plots

def plotScatterMatrix(df, plotSize, textSize):

    df = df.select_dtypes(include =[np.number]) # keep only numerical columns

    # Remove rows and columns that would lead to df being singular

    df = df.dropna('columns')

    df = df[[col for col in df if df[col].nunique() > 1]] # keep columns where there are more than 1 unique values

    columnNames = list(df)

    if len(columnNames) > 10: # reduce the number of columns for matrix inversion of kernel density plots

        columnNames = columnNames[:10]

    df = df[columnNames]

    ax = pd.plotting.scatter_matrix(df, alpha=0.75, figsize=[plotSize, plotSize], diagonal='kde')

    corrs = df.corr().values

    for i, j in zip(*plt.np.triu_indices_from(ax, k = 1)):

        ax[i, j].annotate('Corr. coef = %.3f' % corrs[i, j], (0.8, 0.2), xycoords='axes fraction', ha='center', va='center', size=textSize)

    plt.suptitle('Scatter and Density Plot')

    plt.show()

raw1 = pd.read_csv('../input/steam.csv', delimiter=',')

nRow, nCol = raw1.shape

print(f'There are {nRow} rows and {nCol} columns')
raw1.head(5)
def remove_non_english(df):

    df = df[df['english'] == 1].copy()

    

    # doesn't contain 3 or more non-ascii characters

    df = df[~df['name'].str.contains('[^\u0001-\u007F]{3,}')]

    

    df = df.drop('english', axis=1)

    

    return df





def calc_rating(row):



    import math



    pos = row['positive_ratings']

    neg = row['negative_ratings']



    total_reviews = pos + neg



    average = pos / total_reviews



    score = average - (average*0.5) * 2**(-math.log10(total_reviews + 1))



    return score * 100





def get_unique(series):

    return set(list(itertools.chain(*series.apply(lambda x: [c for c in x.split(';')]))))





def process_cat_gen_tag(df):

    cat_cols = get_unique(df['categories'])

    

    # remove comments to use all categories

    cat_cols = [

#         'Local Multi-Player',

#         'MMO',

#         'Mods',

        'Multi-player',

#         'Online Co-op',

#         'Online Multi-Player',

        'Single-player'

    ]

    

    for col in sorted(cat_cols):

        col_name = re.sub(r'[\s\-\/]', '_', col.lower())

        col_name = re.sub(r'[()]', '', col_name)

        

        df[col_name] = df['categories'].apply(lambda x: 1 if col in x.split(';') else 0)

        

    gen_cols = get_unique(df['genres'])

    

    # remove comments to use all genres. Currently uses main ones as per steam store home page

    gen_cols = [

#         'Accounting',

        'Action',

        'Adventure',

#         'Animation & Modeling',

#         'Audio Production',

        'Casual',

#         'Design & Illustration',

#         'Documentary',

#         'Early Access',

#         'Education',

#         'Free to Play',

#         'Game Development',

#         'Gore',

        'Indie',

        'Massively Multiplayer',

#         'Nudity',

#         'Photo Editing',

        'RPG',

        'Racing',

#         'Sexual Content',

        'Simulation',

#         'Software Training',

        'Sports',

        'Strategy'

#         'Tutorial',

#         'Utilities',

#         'Video Production',

#         'Violent',

#         'Web Publishing'

    ]

    

    gen_col_names = []

    

    for col in sorted(gen_cols):

        col_name = col.lower().replace('&', 'and').replace(' ', '_')

        gen_col_names.append(col_name)

        

        df[col_name] = df['genres'].apply(lambda x: 1 if col in x.split(';') else 0)

    

    # remove "non-games" based on genre

    gen_sums = df[gen_col_names].sum(axis=1)

    df = df[gen_sums > 0]

        

    tag_cols = get_unique(df['steamspy_tags'])

    # df['top_tag'] = df['tags'].apply(lambda x: x.split(';')[0] if ';' in x else x)

    

    # display(cat_cols, gen_cols, tag_cols)

    df = df.drop(['categories', 'genres', 'steamspy_tags'], axis=1)

    

    return df





def process_year(df):

    df['release_date'] = df['release_date'].astype('datetime64[ns]')

    df['release_year'] = df['release_date'].apply(lambda x: x.year)

    return df



    

def process_revenue(df):

    df['est_revenue'] = df['owners'] * df['price']

    return df





def process_price(df):

    cut_points = [-1, 0, 4, 10, 30, 50, 1000]

    label_names = ['free', 'very cheap', 'cheap', 'moderate', 'expensive', 'very expensive']

    

    df['price_categories'] = pd.cut(df['price'], cut_points, labels=label_names)

    

    return df





def pre_process(df):

    # english only

    df = remove_non_english(df)

    

    # windows only

    df = df[df['platforms'].str.contains('windows')].drop('platforms', axis=1).copy()

    

    df['owners'] = df['owners'].str.split('-').apply(lambda x: (int(x[0]) + int(x[1])) / 2).astype(int)

    # df['owners'] = df['owners'].str.replace('[+,]', '').astype(int)

    

    # calculate ratings

    df['total_ratings'] = df['positive_ratings'] + df['negative_ratings']

    df['ratings_ratio'] = df['positive_ratings'] / df['total_ratings']

    df['weighted_rating'] = df.apply(calc_rating, axis=1)

    # df = df.drop(['positive', 'negative'], axis=1)

    

    df = df.rename(columns={'average_forever': 'average_playtime', 'median_forever': 'median_playtime'})

    

    df = process_year(df)

    df = process_revenue(df)

    df = process_price(df)

    df = process_cat_gen_tag(df)

    

    return df



df1 = pre_process(raw1)

df1.dataframeName = 'steam'



nRow, nCol = df1.shape

print(f'There are {nRow} rows and {nCol} columns after preprocessing')



df1.head()
# percentage of apps in each genre

genre_cols = df1.columns[-10:]



total_owners_per_genre = df1[genre_cols].multiply(df1['owners'], axis='index').sum()

average_owners_per_genre = total_owners_per_genre / df1[genre_cols].sum()



fig, ax1 = plt.subplots()



color = 'tab:red'

df1[genre_cols].mean().sort_index(ascending=False).plot.barh(ax=ax1, color=color, alpha=.5, position=.8)

# ax1.set_ylabel('genre')

ax1.set_xlabel('% of apps', color=color)

ax1.tick_params(axis='x', labelcolor=color)



ax2 = ax1.twiny()



color = 'tab:blue'

average_owners_per_genre.sort_index(ascending=False).plot.barh(ax=ax2, color=color, alpha=.5, position=.2, label='1')

ax2.set_xlabel('average owners per app', color=color)

ax2.tick_params(axis='x', labelcolor=color)



plt.tight_layout()

plt.show()
dfsample = df1.sample(1000)

dfsample.dataframeName = 'steam'

plotPerColumnDistribution(dfsample, 10, 5)
plotCorrelationMatrix(dfsample, 8)
plotScatterMatrix(dfsample, 20, 10)