import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



# import os

# for dirname, _, filenames in os.walk('../input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))

jp_df = pd.read_csv("../input/japanesewordsfrequency/japanese_lemmas.csv")

en_df = pd.read_csv("../input/englishwordfrequency/unigram_freq.csv")



jp_df = jp_df[0:100]

en_df = en_df[0:100]



jp_df['word'] = jp_df['lemma']

jp_df["frequency"] = jp_df['frequency'].apply(lambda x: int(x))

jp_df = jp_df[['word', 'frequency', 'rank']]



en_df["frequency"] = (en_df['count'] * (jp_df.iloc[0]["frequency"] / en_df.iloc[0]['count'])).apply(lambda x: int(x))

en_df["rank"] = None

en_df = en_df[['word', 'frequency', 'rank']]



jp_df["Language"] = 'Japanese'

en_df["Language"] = 'English'

jp_df["Word Length"] = jp_df['word'].apply(lambda x: len(x))

en_df["Word Length"] = en_df['word'].apply(lambda x: len(str(x)))

jp_df["Preposition or Joshi"] = 'NO'

en_df["Preposition or Joshi"] = 'NO'

postpositions = pd.read_csv("../input/utf8postposition/postpositions-in-japanese-utf8.csv")

for idx, row in jp_df.iterrows():

    for postposition in postpositions['Word']:

        if row['word'] == str(postposition):

            jp_df.iloc[idx, jp_df.columns.get_loc('Preposition or Joshi')] = 'YES'

jp_df.head()
prepositions = pd.read_csv("../input/prepositions/prepositions-in-english.csv")

for idx, row in en_df.iterrows():

    en_df.iloc[idx, en_df.columns.get_loc('rank')] = idx + 1

    for preposition in prepositions['Word']:

        if row['word'] == preposition:

            en_df.iloc[idx, en_df.columns.get_loc('Preposition or Joshi')] = 'YES'

en_df.head()
df = pd.concat([jp_df, en_df])

g = sns.catplot(x='Word Length', y='frequency', data=df, hue='Preposition or Joshi', kind='swarm', col='Language')

(g.set_axis_labels("Word Length", "Words Frequency")

  .set_titles("{col_name} {col_var} Top 100 Words"))  