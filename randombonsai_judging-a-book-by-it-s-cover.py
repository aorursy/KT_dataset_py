import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import re

import spacy

import numpy

import wordcloud



from nltk.corpus import stopwords

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
nlp = spacy.load('en_core_web_lg')
path = "../input/books.csv"

dfr = pd.read_csv('../input/books.csv', skiprows=[4011, 5687, 7055, 10600, 10667])
dfr.columns
languages = dfr['language_code'].tolist()

l_dict = {}

for l in set(languages):

    l_dict[l] = languages.count(l)

lists = sorted(l_dict.items())

x, y = zip(*lists)

f, ax = plt.subplots(figsize=(12, 5))

plt.barh(x, y)

plt.show()
df = dfr[dfr['ratings_count'] >= 10]

english_codes = ["eng", "en-US", "en-CA", "en-GB"]

df = df[df.language_code.isin(english_codes)]
df["title_length"] = df.apply(lambda row: len(row.title), axis=1)
f, ax = plt.subplots(figsize=(9, 6))

sns.despine(f, left=True, bottom=True)

sns.scatterplot(x=df.title_length, y=df.average_rating,

                hue=df['# num_pages'], 

                palette="ch:r=-.2,d=.3_r",

                sizes=(1, 8), linewidth=0,

                data=df, ax=ax)
f, ax = plt.subplots(figsize=(6.5, 6.5))

sns.despine(f, left=True, bottom=True)

sns.scatterplot(y=df.average_rating, x=df['# num_pages'],

                hue=df.title_length, 

                palette="ch:r=-.2,d=.3_r",

                sizes=(1, 8), linewidth=0,

                data=df, ax=ax)
def pos_in_doc(text, postag):

    doc = nlp(text)

    return 1 if postag in [token.pos_ for token in doc] else 0
df["verb_in_title"] = df.apply(lambda row: pos_in_doc(row.title, "VERB"), axis=1)

df["noun_in_title"] = dfr.apply(lambda row: pos_in_doc(row.title, "NOUN"), axis=1)

df["adv_in_title"] = dfr.apply(lambda row: pos_in_doc(row.title, "ADV"), axis=1)

df["adj_in_title"] = dfr.apply(lambda row: pos_in_doc(row.title, "ADJ"), axis=1)

df["propn_in_title"] = dfr.apply(lambda row: pos_in_doc(row.title, "PROPN"), axis=1)
plt.figure(1, figsize=(10, 6))

plt.subplot(2, 3, 1)

sns.violinplot(x=df.verb_in_title, y=df.average_rating,

               split=True, inner="quart",

               data=df)

plt.subplot(2, 3, 2)

sns.violinplot(x=df.noun_in_title, y=df.average_rating,

               split=True, inner="quart",

               data=df)

plt.subplot(2, 3, 3)

sns.violinplot(x=df.adv_in_title, y=df.average_rating,

               split=True, inner="quart",

               data=df)

plt.subplot(2, 3, 4)

sns.violinplot(x=df.adj_in_title, y=df.average_rating,

               split=True, inner="quart",

               data=df)

plt.subplot(2, 3, 5)

sns.violinplot(x=df.propn_in_title, y=df.average_rating,

               split=True, inner="quart",

               data=df)
print(df['title'][0])
relevant_pos = ['ADJ', 'VERB', 'NOUN', 'PROPN', 'ADV']

col = ["word", "ratings_list", "pos_tag"]

title_words_df = pd.DataFrame(columns=col)

title_words_df = title_words_df.set_index('word')

for index, row in df.iterrows():

    rating = row['average_rating']

    short_title = re.sub(r" \(.*", "", row['title'])

    doc = nlp(short_title)

    tokens = [t for t in doc if t.pos_ in relevant_pos]

    for token in tokens:

        lemma = token.lemma_.lower()

        pos_tag = token.pos_

        if lemma not in title_words_df.index.values:

            mini_df = pd.DataFrame({col[0]: lemma, col[1]: [numpy.array([rating])], col[2]: pos_tag})

            mini_df = mini_df.set_index('word')

            title_words_df = title_words_df.append(mini_df)

        else:

            ex_value = title_words_df.get_value(lemma, col[1])

            new_value = numpy.append(ex_value, [rating])

            title_words_df.at[lemma, col[1]] = new_value
title_words_df["frequency"] = title_words_df.apply(lambda row: len(row['ratings_list']), axis=1)

title_words_df["variance"] = title_words_df.apply(lambda row: numpy.var(row['ratings_list']), axis=1)

title_words_df["mean_rating"] = title_words_df.apply(lambda row: numpy.mean(row['ratings_list']), axis=1)
freq_words_df = title_words_df[title_words_df.frequency > 20]

adj_freq_words_df = freq_words_df[freq_words_df.pos_tag == 'ADJ']

noun_freq_words_df = freq_words_df[freq_words_df.pos_tag == 'NOUN']

adv_freq_words_df = freq_words_df[freq_words_df.pos_tag == 'ADV']

verb_freq_words_df = freq_words_df[freq_words_df.pos_tag == 'VERB']

noun_freq_words_df.sort_values("mean_rating", ascending=False).head(5)
noun_freq_words_df.sort_values("mean_rating", ascending=True).head(5)
stop = set(STOPWORDS)

good_text = " ".join(re.sub("\(.*", "", t) for t in df[df.average_rating >= 4.5].title)

bad_text = " ".join(re.sub("\(.*", "", t) for t in df[df.average_rating <= 3.5].title)

text = " ".join(re.sub("\(.*", "", t) for t in df.title)

good_wordcloud = WordCloud(stopwords=stop, background_color="white").generate(good_text)

bad_wordcloud = WordCloud(stopwords=stop, background_color="white").generate(bad_text)

plt.figure(1, figsize=(14, 10))

plt.subplot(1, 2, 1)

plt.imshow(good_wordcloud, interpolation='bilinear')

plt.axis("off")

plt.subplot(1, 2, 2)

plt.imshow(bad_wordcloud, interpolation='bilinear')

plt.axis("off")
text_wordcloud = WordCloud(stopwords=stop, background_color="white").generate(text)

plt.figure(1, figsize=(14, 10))

plt.imshow(text_wordcloud, interpolation='bilinear')

plt.axis("off")
series_df = df[df.title.str.endswith(')')]
def find_number(text):

    match = re.search(r"#(\d+)", text) 

    if match:

        return int(match.group(1))

    else:

        return 0

    

series_df["n_in_series"] = series_df.apply(lambda row: find_number(row.title), axis=1)
f, ax = plt.subplots(figsize=(10, 6.5))

sns.despine(f, left=True, bottom=True)

sns.scatterplot(y=series_df.average_rating, x=series_df.n_in_series,

                hue=series_df['# num_pages'],

                palette="ch:r=-.2,d=.3_r",

                sizes=(1, 8), linewidth=0,

                data=series_df, ax=ax)
small_series_df = series_df[series_df.n_in_series < 50]

f, ax = plt.subplots(figsize=(10, 6.5))

sns.despine(f, left=True, bottom=True)

sns.scatterplot(y=small_series_df.average_rating, x=small_series_df.n_in_series,

                hue=small_series_df['# num_pages'],

                palette="ch:r=-.2,d=.3_r",

                sizes=(1, 8), linewidth=0,

                data=series_df, ax=ax)