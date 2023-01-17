#Basic

import pandas as pd

import numpy as np

import emoji

import math

import os



#NLP

import nltk

from collections import Counter

from itertools import combinations, islice



#Graphics

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

import wordcloud as wc

from matplotlib.patches import Patch



#ML

from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA

from sklearn.cluster import KMeans
#Read the data

dtypes = {

    'article_host': 'str', 

    'article_name': 'str', 

    'article_subtitle': 'str', 

    'hashtags': 'object',

    'img-label': 'str', 

    'img_src': 'str', 

    'linked_profiles': 'object', 

    'links': 'object', 

    'text': 'str', 

    'text_tokenized': 'object', 

    'text_tokenized_filtered': 'object',

    'text_tokenized_lemmatized': 'object', 

    'page_name': 'category', 

    'page_name_adjusted': 'category',

    'text_length': 'int32', 

    'wc': 'int32', 

    'sc': 'int32',

    'sixltr': 'int32'

}

posts = pd.read_csv('../input/posts_full.csv', index_col=False, dtype=dtypes, 

                    parse_dates=['timestamp']).drop('Unnamed: 0', axis=1)

posts.head()
#Linked_profiles were written as NaN if empty

posts['linked_profiles'] = posts.linked_profiles.fillna('[]')

#Change list columns to lists

list_cols = ['hashtags', 'links', 'linked_profiles', 'text_tokenized', 

             'text_tokenized_filtered', 'text_tokenized_lemmatized']

for col in list_cols:

    posts[col] = posts[col].apply(eval)
#Number of emojis in text

posts['num_emojis'] = posts.text_tokenized.apply(

    lambda x: len([e for e in x if e in emoji.UNICODE_EMOJI]))



#Whether post has an image

posts['has_img'] = ~posts.img_src.isnull()



#Domain from article host

def get_domain(host):

    if type(host) != str:

        return host

    host = host.replace('http://', '').replace('https://', '').split('/')[0]

    if '|' in host:

        host = host.split('|')[0].split('.')[-1].lower()

    elif '.' in host:

        host = host.split('.')[-1].lower()

    else:

        return np.nan

    return host if host.isalpha() else np.nan

posts['article_domain'] = posts.article_host.apply(get_domain).astype('category')



#Fix punc info

puncs = [('periods', '.'), ('exclamations', '!'), ('questionms', '?'), 

         ('equals', '='), ('dollars', '$')]

for name, punc in puncs:

    posts['percent_' + name] = posts.text_tokenized.apply(

        lambda words: words.count(punc)) / posts.num_tokens



#Percent All Caps

posts['percent_all_caps'] = posts.text_tokenized.apply(

    lambda tokens: [token.isupper() for token in tokens].count(True) / 

                    len(tokens) if len(tokens) else 0)
#Scale 'num_' features by number of words to reduce dependence on how long the text is 

skip_percs = {'num_words', 'num_tokens'}

for nc in [n for n in posts.columns if n.startswith('num_') and n not in skip_percs]:

    percent_column_name = 'percent_' + '_'.join(nc.split('_')[1:])

    if percent_column_name not in posts.columns:

        replacement = (posts[nc] / posts.num_words).apply(

            lambda x: x if not math.isinf(x) else 0)

        posts[percent_column_name] = replacement

    posts.drop(nc, axis=1, inplace=True)



#Take log of positive, exponential columns [text_length, num_words, n]

for c in ['text_length', 'num_words', 'num_tokens']:

    posts[c + '_log'] = pd.Series(np.log(posts[c])).replace([np.inf, -np.inf], 0)

    posts.drop(c, axis=1, inplace=True)
#Improve look of graph by cutting spines and adding opaque grid

def pretty_axis(ax, visible_spines=False, y_grid=True, y_grid_alpha=0.4):

    for g in ax.spines:

        ax.spines[g].set_visible(visible_spines)

    ax.yaxis.grid(y_grid, alpha=y_grid_alpha)
#Describe the data

posts.describe()
#Plot numerical data with violin plots

def plot_comparison_violins(df, ncols=3, figsize=None, filename='Violin Comparisons.png', ax_mod=None):

    num = df.select_dtypes(['float64', 'int64', 'int16']).fillna(0)

    num['anti_vax'] = df.anti_vax.apply(

        lambda x: 'Anti-Vax' if x else 'Normal').astype('category')

    nrows = int(np.ceil(len(num.columns[:-1]) / ncols))

    blank = np.zeros(num.shape[0])

    if not figsize:

        figsize=(21, 6 * nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)

    for i, col in enumerate(num.columns[:-1]):

        ax = None

        if ncols * nrows == 1:

            ax = axes

        elif nrows == 1:

            ax = axes[i % ncols]

        else:

            ax = axes[int(i / ncols), i % ncols]

        sns.violinplot(x=blank, y=col, hue='anti_vax', data=num, 

                       split=True, ax=ax, orient='v', legend=False)

        ax.xaxis.set_visible(False)

        ax.legend()

        ax.set_title(col)

        ax.set_ylabel('')

        pretty_axis(ax, y_grid_alpha=0.3)

        if ax_mod:

            ax_mod(ax)

    for i in range(len(num.columns) - 1, nrows * ncols):

        fig.delaxes(axes[int(i / ncols), i % ncols])

    fig.tight_layout()

    fig.savefig(filename, bbox_inches='tight')

plot_comparison_violins(posts[[c for c in posts.columns if 

                               c.startswith('percent_') or c == 'anti_vax'][:12]])
stylistic_columns = ['percent_hashtags', 'percent_linked_profiles', 

                     'percent_links', 'percent_emojis', 'anti_vax']

ax = posts[stylistic_columns].groupby('anti_vax').mean().transpose().plot(

    kind='bar', color=['b', 'r'], figsize=(12, 4))

pretty_axis(ax, y_grid_alpha=0.3)

ax.set_title('Distribution of Stylistic Features')

ax.legend(handles=[Patch(facecolor='red', label='Anti_Vax'), 

                   Patch(facecolor='blue', label='Normal')])

plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right", rotation_mode="anchor")

ax.set_xticklabels(['Percent Hashtags', 'Percent Linked Profiles', 'Percent Links', 'Percent Emojis']);
#Scale numerical data normally

numerical_columns = posts.select_dtypes(['float64', 'float32', 

                                         'int64', 'int32', 'int16']).columns

scaler = StandardScaler()

scaled_features = pd.DataFrame(scaler.fit_transform(posts[numerical_columns].fillna(0)), 

                               columns=numerical_columns)

scaled_features = scaled_features.join(

    posts.select_dtypes(['bool'])).drop('anti_vax', axis=1)

scaled_features.head()
#Run PCA-2 on scaled data for plotting

pca = PCA(n_components=2)

decomposed = pca.fit_transform(scaled_features)

pca_df = pd.DataFrame(decomposed, columns=['x', 'y'])

pca_df['anti_vax'] = posts.anti_vax.apply(

    lambda x: 'Anti-vax' if x else 'Normal').astype('category')



#Plot PCA

plt.figure(figsize=(21, 10), dpi=80)

path = plt.scatter(pca_df.x, pca_df.y, alpha=0.3, cmap='bwr',

                   c=pca_df.anti_vax.apply(lambda x: 'red' if 

                                           x == 'Anti-vax' else 'blue'))

path.axes.set_title('Principle Component Analysis on Two Components');

plt.legend(handles=[Patch(facecolor='red', edgecolor='r', label='Anti-Vax'), 

                    Patch(facecolor='blue', label='Normal')])

for g in path.axes.spines:

    path.axes.spines[g].set_visible(False)

plt.axis('off')

plt.savefig('PCA.png', bbox_inches='tight')
corr = scaled_features.corr()

fig, ax = plt.subplots(figsize=(26, 15))

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True

sns.heatmap(corr, mask=mask, ax=ax, cmap='Spectral', 

            square=True, center=0.0)

ax.set_title('Correlation Heatmap for Numerical Post Data')

fig.tight_layout()

fig.savefig('Correlation Heatmap.png', bbox_inches='tight');
#Show correlation with Anti-Vax

feature_corr = scaled_features.corrwith(posts.anti_vax).abs(

                    ).sort_values(ascending=False).head(40)



#Generate colors for the bars based upon where the features came from

color_categories = {'text_': 'black', 'num_pos': 'orange', 'num_': 'yellow', 

                    'ttr': 'red', 'sentiment_': 'green', 

                    'percent_': 'maroon', 'readability_': 'lightgreen', 

                    'has_': 'coral', 'other': 'blue'}

def generate_categories(col):

    for cc in color_categories:

        if col.startswith(cc):

            return cc

    return 'other'

column_cats = [generate_categories(c) for c in feature_corr.index]

column_colors = [color_categories[c] for c in column_cats]



#Legend for the plot

handles = []

for c in set(column_cats):

    handles.append(Patch(facecolor=color_categories[c], 

                         label=c.replace('_', '')))



fig, ax = plt.subplots(figsize=(21, 15))

feature_corr.plot.bar(ax=ax, width=0.8, rot=75, color=column_colors)

ax.set_ylim((0.15, 0.34))

ax.tick_params(axis='both', which='major', labelsize=15)

ax.set_title('Feature Correlation with Anti-Vax', fontsize=16)

ax.set_ylabel('Correlation')

pretty_axis(ax)

ax.legend(handles=handles, prop={'size': 20})

fig.tight_layout()

fig.savefig('Feature Correlation.png', bbox_inches='tight');
plot_comparison_violins(scaled_features[[feature_corr.index[0]]].join(posts.anti_vax), 

                        ncols=1, filename=feature_corr.index[0] + ' Comparison.png', 

                        figsize=(4, 3), ax_mod=lambda ax: ax.legend(loc='center right'))
cov_pca = PCA(n_components=len(scaled_features.columns))

cov_pca.fit(scaled_features)

pd.set_option('display.float_format', lambda x: '%.5f' % x)

explained_var = pd.Series(cov_pca.explained_variance_ratio_, 

                          index=scaled_features.columns).sort_values(ascending=True)

fig, ax = plt.subplots(figsize=(21, 15))

fig.tight_layout()

ax = explained_var[explained_var > 0.01].plot.barh(ax=ax)

ax.set_title('Explained Variance by Feature')

pretty_axis(ax, y_grid=False)

ax.xaxis.grid(True, alpha=0.3)

ax.yaxis.grid(False)

fig.savefig('Explained Variance.png', bbox_inches='tight');
observing_domains = {'au', 'com', 'ca', 'gov', 'edu', 'net', 'blog', 

                     'org', 'ly', 'uk', 'news', 'ie', 'info', 'tv', 'co'}

domain_posts = posts.loc[~posts.article_domain.isnull(), 

                         ['anti_vax', 'article_domain']]

domain_posts['article_domain'] = domain_posts.article_domain.apply(

    lambda x: x if x in observing_domains else 'other').astype('category')



#Graph the domains

fig, ax = plt.subplots(figsize=(21, 10))

domain_posts = domain_posts.groupby(['anti_vax']).article_domain.value_counts(

                                            ).unstack(fill_value=0).transpose()

domain_posts['sum'] = domain_posts[True] + domain_posts[False]

domain_posts = domain_posts.sort_values(by='sum', 

                                        ascending=False).drop('sum', axis=1)

domain_posts.plot(kind='bar', ax=ax, stacked=False, 

                  color=['blue', 'red'], width=0.9, rot=0)

pretty_axis(ax)

ax.set_xlabel('')

ax.legend(handles=[Patch(facecolor='blue', label='Normal'), 

                   Patch(facecolor='red', label='Anti-Vax')])

ax.set_title('Distribution of Domain Suffixes')

fig.savefig('Domain Distribution.png', bbox_inches='tight')
#Get bigrams of a group posts

stop_words = set(nltk.corpus.stopwords.words("english")).union(

    wc.STOPWORDS).union(observing_domains).union({'http', 'https'})

def bigrams(posts, most_common=20):

    w = posts.text_tokenized_lemmatized.apply(

        lambda words: [word for word in words if word not in stop_words and not word.isdigit()])

    def count_bigrams(words):

        return Counter(zip(words, islice(words, 1, None)))

    return w.apply(count_bigrams).sum().most_common(most_common)

sample_size, most_common = 3000, 50

anti_vax_bigrams = bigrams(posts[posts.anti_vax].sample(n=sample_size), 

                           most_common=most_common)

normal_bigrams = bigrams(posts[~posts.anti_vax].sample(n=sample_size), 

                         most_common=most_common)
#Show Anti_Vax Bigrams

wordcloud = wc.WordCloud(stopwords=stop_words, background_color='white', 

                         width=2100, height=600).fit_words({' '.join(couple): count for 

                                                            couple, count in anti_vax_bigrams})

fig = plt.figure(figsize=(21, 6))

plt.imshow(wordcloud)

plt.axis('off')

fig.savefig('Anti-Vax Bigram Wordcloud.png', bbox_inches='tight')

plt.show()
#Show Normal Bigrams

wordcloud = wc.WordCloud(stopwords=stop_words, background_color='white', 

                         width=2100, height=600).fit_words({' '.join(couple): count for 

                                                            couple, count in normal_bigrams})

fig = plt.figure(figsize=(21, 6))

plt.imshow(wordcloud)

plt.axis('off')

fig.savefig('Normal Bigram Wordcloud.png', bbox_inches='tight')

plt.show()
#Get pairs for a group of posts

def pairs(posts, most_common=20):

    w = posts.text_tokenized_lemmatized.apply(

        lambda words: [word for word in words if 

                       word not in stop_words and not word.isdigit()])

    def pairs(words):

        return list(combinations(set(words), 2))

    counter = Counter()

    for ind, val in w.apply(pairs).iteritems():

        counter.update(val)

    return counter.most_common(most_common)

sample_size = 30000

anti_vax_pairs = pairs(posts[posts.anti_vax].sample(n=sample_size), most_common=None)

normal_pairs = pairs(posts[~posts.anti_vax].sample(n=sample_size), most_common=None)
#Clean pairs and convert to a dataframe

pairs = {}

for anti in anti_vax_pairs:

    a, b = anti[0]

    if b < a:

        a, b = b, a

    pairs[(a, b)] = [anti[1], 0]

for norm in normal_pairs:

    a, b = norm[0]

    if b < a:

        a, b = b, a

    if (a, b) not in pairs:

        pairs[(a, b)] = [0, 0]

    pairs[(a, b)] = [pairs[(a, b)][0], norm[1]]

pairs = pd.DataFrame([(pair, count[0], count[1]) for pair, count in pairs.items()], columns=['Pair', 'Anti-Vax Count', 'Normal Count'])

pairs['Difference'] = pairs['Anti-Vax Count'] - pairs['Normal Count']

pairs['Abs_Difference'] = pairs.Difference.abs()

pairs.set_index('Pair', inplace=True)

pairs.head()
#View Anti-Vax heavy pairs

fig, axes = plt.subplots(3, 1, figsize=(21, 10))

pair_plots = {

    'Anti-Vax Dominant Pairs': pairs.sort_values(['Difference', 'Anti-Vax Count'], 

                                                 ascending=False).iloc[:15, 0], 

    'Normal Dominant Pairs': pairs.sort_values(['Difference', 'Normal Count'], 

                                               ascending=[True, False]).iloc[:15, 1],

    'Common Pairs': pairs[(pairs['Anti-Vax Count'] != 0) & 

                          (pairs['Normal Count'] != 0)].sort_values(['Abs_Difference', 

                                                                     'Normal Count'], 

                                                                    ascending=[True, False]).iloc[:40, :2]

}

for i, ((title, data), color) in enumerate(zip(pair_plots.items(), ['r', 'b', ['r', 'b']])):

    data.plot(kind='bar', ax=axes[i], color=color)

    pretty_axis(axes[i])

    axes[i].set_title(title)

    axes[i].set_xlabel('')

    axes[i].set_ylabel('Count')

    plt.setp(axes[i].xaxis.get_majorticklabels(), rotation=30, ha="right", 

             rotation_mode="anchor") 

fig.tight_layout()

fig.savefig('Pairs Comparison.png', bbox_inches='tight');
from scipy.stats import ttest_ind

def test_significant_scores(df):

    anti_vax, normal = df[df.anti_vax], df[~df.anti_vax]

    data = pd.DataFrame({'Feature': df.columns})

    data['t_value'] = data.Feature.apply(lambda f: ttest_ind(anti_vax[f], normal[f], equal_var=False))

    data['p_value'] = data.t_value.apply(lambda x: x[1])

    data['t_value'] = data.t_value.apply(lambda x: x[0])

    return data

feature_tests = test_significant_scores(scaled_features.join(posts.anti_vax))

feature_tests['Significant'] = feature_tests.p_value <= 0.05

feature_tests.set_index('Feature', inplace=True)

feature_tests.to_csv('Significance.csv')

feature_tests.head()
feature_tests[feature_tests.p_value > 0.05].sort_values(by='p_value', ascending=False)
feature_tests[feature_tests.p_value <= 0.05].sort_values(by='t_value', ascending=True).head()
#Add categorical dummy variables for which hashtags were in the text

#hashtags = set(posts.hashtags.sum())

#for hashtag in hashtags:

#    scaled_features['hashtag_' + hashtag] = posts.hashtags.apply(lambda x: hashtag in x)

#Add categorical dummy variables for the domain of the article linked

#scaled_features = scaled_features.join(pd.get_dummies(posts.article_domain))

#TODO: Add Bigram dummies