import pandas as pd

from tqdm import tqdm

from collections import Counter, defaultdict

from gensim.models.word2vec import Word2Vec

from nltk.tokenize import word_tokenize

import spacy

import numpy as np

import pylab as pl

import seaborn as sns

import random

import math



sns.set()

tqdm.pandas()



en = spacy.load('en')



df = pd.read_csv('../input/17k-apple-app-store-strategy-games/appstore_games.csv')

print(df.columns)
def is_nan(value):

    try:

        return math.isnan(value)

    except TypeError:

        return False



df['In-app Purchases'] = df['In-app Purchases'].apply(lambda r: r if is_nan(r) else [float(i) for i in r.split(', ')])

pd.to_numeric(df['Average User Rating'])



df['name_doc'] = df.Name.progress_apply(lambda r: en(r.replace('\\n', ' ')))

df['name_clean'] = df.name_doc.progress_apply(lambda r: [str(i).lower() for i in r if not (i.is_stop or i.is_punct)])



df['desc_doc'] = df.Description.progress_apply(lambda r: en(r.replace('\\n', ' ')))

df['desc_clean'] = df.desc_doc.progress_apply(lambda r: [str(i).lower() for i in r if not (i.is_stop or i.is_punct)])





print(df[['Name', 'name_clean']].head(10))



df['price_in_app'] = df.apply(lambda r: 1.0 if r['Price'] and not is_nan(r['In-app Purchases']) else np.nan, axis=1)

# somehow sometimes NaN is loaded as float-NaN and sometimes as np.NaN. math.isnan is consistent

mapping = {

    'Average User Rating': ('rating', None),

    'Price': ('price', lambda x: x <= 0.0),

    'In-app Purchases': ('in_app', None),

    'User Rating Count': ('rate_count', None),

    'price_in_app': ('price_in_app', None),

}





def update_sdict(_d, words, value, key, cutoff_rule=None):

    if is_nan(value):

        return

    if str(value).lower() == 'nan':

        return

    if cutoff_rule is not None and cutoff_rule(value):

        return

    try:

        value = max(value)

    except TypeError:

        pass

    for w in words:

        if value is np.nan:

            print(value)

        _d[w][key].append(value)



def make_aggreg_df(df, mapping, word_source='name_clean'):

    x = defaultdict(lambda: {v[0]: [] for v in mapping.values()})

    c = Counter()

    null = df[word_source].apply(lambda r: c.update(set(r)))

    for k, v in c.items():

        x[k]['nb'] = v

    for key in mapping.keys():

        null = df.apply(lambda r: update_sdict(x, set(r[word_source]), r[key], mapping[key][0], mapping[key][1]), axis=1)

    for k, v in x.items():

        v['word'] = k

    agg_df = pd.DataFrame([r for w, r in x.items()])

    for v in mapping.values():

        agg_df['nb_%s' % v[0]] = agg_df[v[0]].apply(len)

        agg_df['avg_%s' % v[0]] = agg_df[v[0]].apply(np.mean)

        agg_df['std_%s' % v[0]] = agg_df[v[0]].apply(np.std)

    agg_df['nb_price_only'] = agg_df.nb_price - agg_df.nb_price_in_app

    agg_df['nb_in_app_only'] = agg_df.nb_in_app - agg_df.nb_price_in_app

    agg_df['nb_free'] = agg_df.nb - agg_df.nb_price - agg_df.nb_in_app + agg_df.nb_price_in_app

    agg_df['nb_pay'] = agg_df.nb_price + agg_df.nb_in_app - agg_df.nb_price_in_app

    return agg_df



agg_df = make_aggreg_df(df, mapping)

print(agg_df[['word', 'nb', 'nb_price_only', 'nb_in_app_only', 'nb_price_in_app']].head(10))

sub = agg_df.sort_values(by='nb', ascending=False)

sub = sub[sub.nb > 50]

sub.set_index('word', inplace=True)

flatui = ["#0974f6", "#f6b709", "#f67809", "#f64009"]

cmap = sns.color_palette(flatui, 4)
f, (a0, a1) = pl.subplots(2, 1, gridspec_kw={'height_ratios': [1, 5]})

sub[['nb_free', 'nb_in_app_only', 'nb_price_only', 'nb_price_in_app']].div(sub.nb, axis='index').head(30).plot(

    kind='bar',

    stacked=True,

    color=cmap,

    figsize=(20, 16),

    ax=a1



)

sub['nb'].head(30).plot(

    kind='bar',

    ax=a0

)

a1.set_xlabel('Most commonly found words')

a1.set_ylabel('nb of games, per category (stacked)')

a0.set_ylabel('word occurences')

a1.legend(['100% free', 'in-app purchases only', 'buy only', 'buy AND in-app purchases'])

pl.suptitle('Game monetization for the most common words found in the game title - normalized')
ax = sub[['nb_free', 'nb_in_app_only', 'nb_price_only', 'nb_price_in_app']].div(sub.nb, axis='index').sort_values(by='nb_free', ascending=False).head(30).plot(

    kind='bar',

    stacked=True,

    color=cmap,

    figsize=(20, 16),

    title='Frequent title words the most associated with a TRUE free game - normalized',

)

ax.set_xlabel('Frequent title words sorted by FREE ratio')

ax.set_ylabel('monetization type proportion')

ax.legend(['100% free', 'in-app purchases only', 'buy only', 'buy AND in-app purchases'])
ax = sub[['nb_free', 'nb_in_app_only', 'nb_price_only', 'nb_price_in_app', 'nb_pay']].div(sub.nb, axis='index').sort_values(by='nb_pay', ascending=False).head(50)[['nb_free', 'nb_price_only', 'nb_in_app_only', 'nb_price_in_app']].head(30).plot(

    kind='bar',

    stacked=True,

    color=cmap,

    figsize=(20, 16),

    title='Frequent title words the most associated with a pay for game - normalized',

)

ax.set_xlabel('Frequent title words sorted by all NON-FREE ratio')

ax.set_ylabel('monetization type proportion')

ax.legend(['100% free', 'in-app purchases only', 'buy only', 'buy AND in-app purchases'])
ax = sub[['nb_free', 'nb_in_app_only', 'nb_price_only', 'nb_price_in_app']].head(30).plot(

    kind='bar',

    stacked=True,

    color=cmap,

    figsize=(20, 16),

    title='Game monetization for the most common words found in the game title',

)

ax.set_xlabel('Most commonly found words')

ax.set_ylabel('nb of games, per category (stacked)')

ax.legend(['100% free', 'in-app purchases only', 'buy only', 'buy AND in-app purchases'])
fig, axes = pl.subplots(nrows=3, ncols=3, figsize=(20, 16))

sub[['nb_free', 'nb_price_only', 'nb_in_app_only', 'nb_price_in_app']].div(sub.nb, axis='index').sort_values(by='nb_free', ascending=False).head(10).plot(kind='bar', stacked=True, color=cmap, ax=axes[0,0])

sub[['nb_free', 'nb_price_only', 'nb_in_app_only', 'nb_price_in_app']].head(10).plot(kind='bar', stacked=True, color=cmap, ax=axes[0,1])

sub[['nb']].head(10).plot(kind='bar', stacked=True, ax=axes[0,2])

sub[['nb_free', 'nb_price_only', 'nb_in_app_only', 'nb_price_in_app']].div(sub.nb, axis='index').sort_values(by='nb_price_only', ascending=False).head(10).plot(kind='bar', stacked=True, color=cmap, ax=axes[1,0])

sub[['nb_free', 'nb_price_only', 'nb_in_app_only', 'nb_price_in_app']].div(sub.nb, axis='index').sort_values(by='nb_in_app_only', ascending=False).head(10).plot(kind='bar', stacked=True, color=cmap, ax=axes[1,1])

sub[['nb_free', 'nb_price_only', 'nb_in_app_only', 'nb_price_in_app']].div(sub.nb, axis='index').sort_values(by='nb_price_in_app', ascending=False).head(10).plot(kind='bar', stacked=True, color=cmap, ax=axes[1,2])

sub[['nb_free', 'nb_price_only', 'nb_in_app_only', 'nb_price_in_app', 'nb_pay']].div(sub.nb, axis='index').sort_values(by='nb_pay', ascending=False).head(10)[['nb_free', 'nb_pay']].plot(kind='bar', stacked=True, color=cmap, ax=axes[2,0])

sub[['nb_free', 'nb_price_only', 'nb_in_app_only', 'nb_price_in_app', 'nb_price']].div(sub.nb, axis='index').sort_values(by='nb_price', ascending=False).head(10)[['nb_free', 'nb_price', 'nb_in_app_only']].plot(kind='bar', stacked=True, color=cmap, ax=axes[2,1])

sub[['nb_free', 'nb_price_only', 'nb_in_app_only', 'nb_price_in_app', 'nb_in_app']].div(sub.nb, axis='index').sort_values(by='nb_in_app', ascending=False).head(10)[['nb_free', 'nb_price_only', 'nb_in_app']].plot(kind='bar', stacked=True, color=cmap, ax=axes[2,2])

class MarkovTextGenerator:

    def __init__(self):

        self.model = None



    def train(self, texts):

        model = defaultdict(Counter)

        STATE_LEN = 2

        for text in tqdm(texts):

            for i in range(len(text) - STATE_LEN):

                state = text[i:i + STATE_LEN]

                next = text[i + STATE_LEN]

                model[' '.join(state)][next] += 1

        self.model = model



    def run(self, max_len=25):

        state = random.choice(list(self.model)).split()

        out = list(state)

        for i in range(max_len):

            try:

                x = random.choices(list(self.model.get(' '.join(state))), self.model.get(' '.join(state)).values())

            except TypeError:

                break

            out.extend(x)

            state = state[1:]

            state.append(out[-1])

        return ' '.join(out)



    

model = MarkovTextGenerator()

model.train(df.desc_doc.progress_apply(lambda r: [str(i).lower() for i in r]))



for i in range(10):

    print(model.run(50))

    print()
