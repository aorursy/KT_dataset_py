from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt # plotting

import numpy as np # linear algebra

import os # accessing directory structure

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
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

nRowsRead = None # specify 'None' if want to read whole file

# data.csv may have more rows in reality, but we are only loading/previewing the first 1000 rows

df1 = pd.read_csv('/kaggle/input/data.csv', delimiter=',', nrows = nRowsRead)

df1.dataframeName = 'data.csv'

nRow, nCol = df1.shape

print(f'There are {nRow} rows and {nCol} columns')
df1.head(5)
plotPerColumnDistribution(df1, 10, 5)
plotCorrelationMatrix(df1, 8)
plotScatterMatrix(df1, 20, 10)
df = df1.copy()

df = df[df['published_date'] < '2020-01-23'] # an article age more than 1 month (to stable ratings)

df['published_date'] = pd.to_datetime(df['published_date'])

df['published_dayofweek'] = df['published_date'].dt.dayofweek # 0 - Monday

df['published_dayname'] = df['published_date'].dt.day_name()

df['published_year'] = df['published_date'].dt.year

df['published_hour'] = df['published_time'].apply(lambda x: int(x[:2]))

print(df.shape)

df[['link', 'title', 'published_date', 'published_time']].head()
plt.rcParams["figure.figsize"] = (10, 6)
sample = df[['published_dayofweek', 'published_dayname', 'rating', 'comments', 'views']].groupby(['published_dayofweek', 'published_dayname']).agg('mean').reset_index()

width = 0.35

x = np.arange(len(sample))

fig, ax = plt.subplots()

rects1 = ax.bar(x - width/2, sample['rating'], width, label='rating')

rects2 = ax.bar(x + width/2, sample['comments'], width, label='comments')

ax.set_ylabel('stats')

ax.set_title('Counters by day of week')

ax.set_xticks(x)

ax.set_xticklabels(sample['published_dayname'])

ax.legend()

plt.show()
sample = df[['published_hour', 'rating', 'comments', 'views']].groupby(['published_hour']).agg('mean').reset_index()

width = 0.35

x = np.arange(len(sample))

fig, ax = plt.subplots()

rects1 = ax.bar(x - width/2, sample['rating'], width, label='rating')

rects2 = ax.bar(x + width/2, sample['comments'], width, label='comments')

ax.set_ylabel('stats')

ax.set_title('Counters by hour')

ax.set_xticks(x)

ax.set_xticklabels(sample['published_hour'])

ax.legend()

plt.show()
sample = df[['published_year', 'rating', 'comments']].groupby(['published_year']).agg('mean').reset_index()

width = 0.35

x = np.arange(len(sample))

fig, ax = plt.subplots()

rects1 = ax.bar(x - width/2, sample['rating'], width, label='rating')

rects2 = ax.bar(x + width/2, sample['comments'], width, label='comments')

ax.set_ylabel('stats')

ax.set_title('Counters by year')

ax.set_xticks(x)

ax.set_xticklabels(sample['published_year'])

ax.legend()

plt.show()
# it's some approx, because it doesn't contain all articles

sample = df[['published_year', 'rating', 'comments']].groupby(['published_year']).agg('sum').reset_index()

width = 0.35

x = np.arange(len(sample))

fig, ax = plt.subplots()

rects1 = ax.bar(x - width/2, sample['rating'], width, label='rating')

rects2 = ax.bar(x + width/2, sample['comments'], width, label='comments')

ax.set_ylabel('stats')

ax.set_title('Activity by year')

ax.set_xticks(x)

ax.set_xticklabels(sample['published_year'])

ax.legend()

plt.show()
from collections import Counter
c = Counter(', '.join(list(df['article_categories'][~df['article_categories'].isnull()])).split(', '))

most_common_article_categs = c.most_common(10)

most_common_article_categs
c = Counter(', '.join(list(df['tags'][~df['tags'].isnull()])).split(', '))

most_common_tags = c.most_common(10)

most_common_tags
plt.hist(df['views'], bins=100, color='#0504aa')

plt.show()
plt.hist(np.log(df['views']+1), bins=100, color='#0504aa')

plt.show()
plt.hist(np.log(df['sentences_count']+1), bins=100, color='#0504aa')

plt.show()
df_sample = df[(df['positive_votes'] + df['negative_votes']) > 10].copy()

print(len(df_sample))

df_sample['rating_2'] = df_sample['positive_votes'] / (df_sample['positive_votes'] + df_sample['negative_votes'])
sample = df_sample[['published_dayofweek', 'published_dayname', 'rating_2']].groupby(['published_dayofweek', 'published_dayname']).agg('mean').reset_index()

plt.bar(range(len(sample)), sample['rating_2'])

plt.xticks(range(len(sample)), sample['published_dayname'])

plt.title('Rating_2 by dayofweek')

plt.ylabel('stats')

plt.show()
plt.hist(df_sample['rating_2'], bins=100, color='#0504aa')

plt.title('rating_2 distrib')

plt.xlabel('rating_2')

plt.ylabel('count')

plt.show()
df_sample['rating_2_round'] = df_sample['rating_2'].apply(lambda x: round(x*10)/10)

sample = df_sample[['rating_2_round', 'sentences_count']].groupby('rating_2_round').agg('mean').reset_index()



plt.bar(range(len(sample)), sample['sentences_count'])

plt.xticks(range(len(sample)), sample['rating_2_round'])

plt.title('Sent length rating_2 dependency')

plt.xlabel('rating_2')

plt.ylabel('sentences count')

plt.show()
most_common_article_categs = set([x[0] for x in most_common_article_categs])

most_common_tags = set([x[0] for x in most_common_tags])

most_common_article_categs, most_common_tags
def is_article_categ_most_common(article_categ_line):

    if pd.isnull(article_categ_line):

        return 0

    categs = article_categ_line.split(', ')

    return int(any([categ in most_common_article_categs for categ in categs]))



def is_tag_most_common(tags_line):

    if pd.isnull(tags_line):

        return 0

    tags = tags_line.split(', ')

    return int(any([tag in most_common_tags for tag in tags]))
df_sample['title_len'] = df_sample['title'].apply(len)

df_sample['article_categ_cnt'] = df_sample['article_categories'].apply(lambda x: x.count(',') if pd.notnull(x) else 0)

df_sample['tags_cnt'] = df_sample['tags'].apply(lambda x: x.count(',') if pd.notnull(x) else 0)

df_sample['weekend'] = df_sample['published_dayofweek'].apply(lambda x: int(x in [5, 6]))

df_sample['most_common_article_categ'] = df_sample['article_categories'].apply(is_article_categ_most_common)

df_sample['most_common_tag'] = df_sample['tags'].apply(is_tag_most_common)

df_sample['weight'] = df_sample['negative_votes'] + df_sample['positive_votes']
features = [

    'title_len', 'article_categ_cnt', 'href_count', 'img_count', 'tags_cnt', 'h3_count',

    'i_count', 'spoiler_count', 'text_len', 'lines_count', 'sentences_count',

    'max_sentence_len', 'min_sentence_len', 'mean_sentence_len', 'median_sentence_len',

    'tokens_count', 'max_token_len', 'mean_token_len', 'median_token_len',

    'alphabetic_tokens_count', 'words_count', 'words_mean', 'published_dayofweek',

    'published_hour', 'weekend', 'most_common_article_categ', 'most_common_tag',

    'weight'

]



y = df_sample['rating_2']

X = df_sample[features]

y.shape, X.shape
plt.hist(np.log(1.1-y), bins=40)

plt.show()
from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

train_weights = X_train['weight']

X_train.drop(['weight'], axis=1, inplace=True)

X_test.drop(['weight'], axis=1, inplace=True)
import lightgbm as lgb
gbm = lgb.LGBMRegressor(num_leaves=31, learning_rate=0.04, n_estimators=100)



gbm.fit(X_train, np.log(1.1 - y_train),

        sample_weight=train_weights,

        eval_set=[(X_test, np.log(1.1 - y_test))],

        eval_metric='l2', early_stopping_rounds=20)

y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration_)
mean_squared_error(y_test, 1.1 - np.exp(y_pred)) ** 0.5
plt.hist(1.1 - np.exp(y_pred), bins=10, label='pred')

plt.hist(y_test, bins=10, label='true', alpha=0.8)

plt.show()
np.corrcoef(y_test, 1.1 - np.exp(y_pred))
sample = X_test.copy()

sample['pred'] = y_pred

sample = sample.sort_values('pred')
articles_cnt = 10

best_articles = [row for _, row in df_sample.loc[list(sample.index[:articles_cnt])].iterrows()]

worst_articles = [row for _, row in df_sample.loc[list(sample.index[-articles_cnt:])].iterrows()]



print('best articles')

for article in best_articles:

    print(f'    {article["title"]}  {article["positive_votes"]}  {article["negative_votes"]}\n    {article["rating_2"]}\n    {article["link"]}')

print('worst articles')

for article in worst_articles:

    print(f'    {article["title"]}  {article["positive_votes"]}  {article["negative_votes"]}\n    {article["rating_2"]}\n    {article["link"]}')