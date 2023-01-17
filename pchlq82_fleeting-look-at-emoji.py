!pip install -q demoji
import numpy as np 

import pandas as pd 

import os

from joblib import Parallel, delayed

import time

from sklearn.preprocessing import LabelEncoder

from collections import ChainMap, Counter

import itertools



from wordcloud import WordCloud

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
import emoji

import demoji

demoji.download_codes()
path_data = '../input/jigsaw-multilingual-toxic-comment-classification/'

cols_to_use = ['id', 'comment_text', 'toxic']



train_tc = pd.read_csv(path_data + 'jigsaw-toxic-comment-train.csv', usecols=cols_to_use)
# encoding id to save memory

train_tc.id = LabelEncoder().fit_transform(train_tc.id)
train_tc.head()
def find_emoji(idx, row):

    if demoji.findall(row):

        return { idx: [v for _, v in demoji.findall(row).items()] }
%%time

parallel =  Parallel(n_jobs=-1, backend='multiprocessing', verbose=0)

joblist = [ delayed(find_emoji)(idx, row) for idx, row in zip(train_tc.id.values, 

                                                              train_tc.comment_text.values) if delayed(find_emoji)(idx, row) ]

retlist  =  parallel( joblist )

list_of_dicts = list(filter(lambda x: type(x)==dict, retlist))

d = dict(ChainMap(*list_of_dicts))
train_tc['emoji_cnt'] = train_tc.id.map( {k:len(v) for k, v in d.items()} ).fillna(0).astype(int)

train_tc['emoji'] = train_tc.id.map(d).fillna('none')
sns.countplot( train_tc.emoji_cnt[train_tc.emoji_cnt != 0], hue=train_tc.toxic)

plt.title("Number of emojis in comment text");
train_tc['emoji_cnt'] = train_tc[train_tc.emoji != 'none'].emoji.apply( lambda x: len(x) )
_, ax = plt.subplots(figsize=(15,5)) 

ax = sns.heatmap( pd.crosstab(train_tc.emoji_cnt, train_tc.toxic), cmap="Blues", fmt='g', annot=True, cbar=False )

plt.title("Number of emojis in comment text");
non_toxic_emoji = list(itertools.chain(*train_tc[(train_tc.emoji != 'none') & (train_tc.toxic == 0)].emoji.values))

toxic_emoji = list(itertools.chain(*train_tc[(train_tc.emoji != 'none') & (train_tc.toxic == 1)].emoji.values))
c0 = Counter()

c1 = Counter()

for emoji_non_toxic, emoji_toxic in zip(non_toxic_emoji, toxic_emoji):

    c0[emoji_non_toxic] += 1

    c1[emoji_toxic] += 1
print("Most common non toxic:", c0.most_common(5))

print()

print("Most common toxic:", c1.most_common(5))
def concat_items(lst):

    return list(map(lambda x: x.replace(' ', '_'), lst))



text0 = ' '.join(concat_items(non_toxic_emoji))

text1 = ' '.join(concat_items(toxic_emoji))
def plot_wordcloud(text, title, toxic=True):

    

    back_color = "grey" if toxic else "white"

    wordcloud = WordCloud(relative_scaling=1.0, width=1000, 

                          height=1000, max_font_size=100, 

                          max_words=100, background_color=back_color).generate(text)

    plt.figure(figsize=(15,10))

    plt.imshow(wordcloud, interpolation="bilinear")

    plt.title(title, fontsize= 30)

    plt.axis("off")

    plt.show()
plot_wordcloud(text=text1, title="Toxic emojis", toxic=True)
plot_wordcloud(text=text0, title="Non toxic emojis", toxic=False)