import numpy as np

import pandas as pd

from tqdm import tqdm

from nltk.tokenize import word_tokenize

import nltk

from nltk.stem import WordNetLemmatizer

from multiprocessing import Pool

from nltk.corpus import wordnet

import pylab as pl

import seaborn as sns



tqdm.pandas()
df = pd.read_csv('/kaggle/input/1-million-reddit-comments-from-40-subreddits/kaggle_RC_2019-05.csv')
lemmatizer = WordNetLemmatizer()



def worker_tagging(text):

    return nltk.pos_tag(word_tokenize(text))



with Pool(9) as p:

    # NOTE: if you run this code yourself, tqdm is strugging a bit to update with the multiprocessing

    # Do not worry if the progress bar gets stuck, it updates by BIG increments

    tagged = p.map(worker_tagging, tqdm(list(df.body)))



df['tagged'] = tagged
def get_wordnet_pos(treebank_tag):

    # nltk taggers uses the treebank tag format, but WordNetLemmatizer needs a different \ simpler format

    if treebank_tag.startswith('J'):

        return wordnet.ADJ

    elif treebank_tag.startswith('V'):

        return wordnet.VERB

    elif treebank_tag.startswith('N'):

        return wordnet.NOUN

    elif treebank_tag.startswith('R'):

        return wordnet.ADV

    else:

        return ''



def lemmatize(lemmatizer, token_tag):

    lemmas = []

    for i, j in token_tag:

        j = get_wordnet_pos(j)

        try:

            lemmas.append(lemmatizer.lemmatize(i, j))

        except KeyError:  # some tags are not known by WordNetLemmatizer

            lemmas.append(i)

    return lemmas



df['tokens'] = df.tagged.progress_apply(lambda r: lemmatize(lemmatizer, r))

from collections import Counter, defaultdict



CATEGORY_HEADER = 'subreddit'

tokens_counter = Counter()

category_tokens_counter = defaultdict(Counter)

_ = df.progress_apply(lambda r: (tokens_counter.update(r['tokens']), category_tokens_counter[r[CATEGORY_HEADER]].update(r['tokens'])), axis=1)

nb_tokens = sum(tokens_counter.values())

nb_token_per_category = {category: sum(c.values()) for category, c in category_tokens_counter.items()}
TOPN_TOKENS = 3000

print('keeping %s tokens out of %s' % (TOPN_TOKENS, nb_tokens))



def representativeness(token, tokens_counter, category_tokens_counter, nb_tokens, nb_token_per_categ):

    representativeness_scores = {

        categ: category_tokens_counter.get(categ).get(token, 0) / tokens_counter.get(token) * nb_tokens / nb_token_per_categ[categ] for categ in category_tokens_counter.keys()

    }

    representativeness_scores['token'] = token

    representativeness_scores['token_count'] = tokens_counter.get(token)

    return representativeness_scores



representativeness_df = pd.DataFrame([representativeness(x[0], tokens_counter, category_tokens_counter, nb_tokens, nb_token_per_category) for x in tokens_counter.most_common(TOPN_TOKENS)])

representativeness_df.sort_values(by='token_count', inplace=True, ascending=False)
BAN_SET = {'/','*','^'}



def ban_token(token, ban_set):

    return bool(set(token).intersection(ban_set))



representativeness_df['ban'] = representativeness_df.token.apply(lambda r: ban_token(r, BAN_SET))

representativeness_df = representativeness_df[representativeness_df.ban == False]

representativeness_df = representativeness_df.set_index('token')
TOPN_PER_SUB = 12

MAX_VISUAL_TOKEN_LEN = 12



fig, axes = pl.subplots(7, 6, figsize=(16, 18), dpi=80, facecolor='w', edgecolor='k')

axes = axes.flatten()

[pl.setp(ax.get_xticklabels(), rotation=90) for ax in axes]

pl.subplots_adjust(wspace=0.8)

pl.subplots_adjust(hspace=0.4)

for i, subreddit in enumerate(category_tokens_counter.keys()):

    sorted_scores = representativeness_df[subreddit].sort_values()

    topn = sorted_scores.tail(TOPN_PER_SUB)

    xlabels = [i if len(i) < MAX_VISUAL_TOKEN_LEN else i[:MAX_VISUAL_TOKEN_LEN-2] + '..' for i in topn.index ]

    sns.barplot(topn.values, xlabels, ax=axes[i])

    axes[i].set_title(subreddit)

pl.title("Most over used words per subreddit (top %s words)" % TOPN_TOKENS)

fig, axes = pl.subplots(7, 6, figsize=(16, 18), dpi=80, facecolor='w', edgecolor='k')

axes = axes.flatten()

[pl.setp(ax.get_xticklabels(), rotation=90) for ax in axes]

fig.tight_layout()

for i, subreddit in enumerate(category_tokens_counter.keys()):

    sorted_scores = representativeness_df[subreddit].sort_values()

    representativeness_df[subreddit].hist(ax=axes[i], bins=100)

    axes[i].set_title(subreddit)

    axes[i].set_yscale('log')

    axes[i].set_xlim(0, 60)

pl.title("Representativeness distribution / subreddit")