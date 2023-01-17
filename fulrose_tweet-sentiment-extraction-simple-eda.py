import pandas as pd

import numpy as np

import os

import seaborn as sns

import matplotlib.pyplot as plt



DATA_PATH = "../input/tweet-sentiment-extraction"

os.listdir(DATA_PATH)
df_train = pd.read_csv(os.path.join(DATA_PATH, 'train.csv'))

df_test = pd.read_csv(os.path.join(DATA_PATH, 'test.csv'))
df_train["text_size"] = df_train["text"].map(lambda x: len(str(x)))

df_train["selected_text_size"] = df_train["selected_text"].map(lambda x: len(str(x)))

df_train["text_selected_ratio"] = df_train["selected_text_size"] / df_train["text_size"]



df_test["text_size"] = df_test["text"].map(lambda x: len(str(x)))
df_train.head()
df_test.head()
fig, ax = plt.subplots(1, 1, figsize=(7, 7))

plt.title("Number of sentiments for train set", fontsize=20)

sns.countplot(df_train["sentiment"], ax=ax)

plt.show()
fig, ax = plt.subplots(1, 1, figsize=(7, 7))

plt.title("Number of sentiments for test set", fontsize=20)

sns.countplot(df_test["sentiment"], ax=ax)

plt.show()
fig, ax = plt.subplots(1,3, figsize=(15, 5))

sns.distplot(df_train["text_size"], ax=ax[0])

sns.distplot(df_train["selected_text_size"], ax=ax[1])

sns.distplot(df_train["text_selected_ratio"], ax=ax[2])

plt.show()
fig, ax = plt.subplots(1, 3, figsize=(15, 5))

sns.distplot(df_train.loc[df_train["sentiment"]=='neutral', 'text_size'].rename("neutral"), label='text', ax=ax[0])

ax[0].legend()

sns.distplot(df_train.loc[df_train["sentiment"]=='positive', 'text_size'].rename("positive"), label='text', ax=ax[1])

ax[1].legend()

sns.distplot(df_train.loc[df_train["sentiment"]=='negative', 'text_size'].rename("negative"), label='text', ax=ax[2])

ax[2].legend()

plt.show()
fig, ax = plt.subplots(1, 3, figsize=(15, 5))

sns.distplot(df_train.loc[df_train["sentiment"]=='neutral', 'selected_text_size'].rename("neutral"), label='selected', ax=ax[0])

ax[0].legend()

sns.distplot(df_train.loc[df_train["sentiment"]=='positive', 'selected_text_size'].rename("positive"), label='selected', ax=ax[1])

ax[1].legend()

sns.distplot(df_train.loc[df_train["sentiment"]=='negative', 'selected_text_size'].rename("negative"), label='selected', ax=ax[2])

ax[2].legend()

plt.show()
fig, ax = plt.subplots(1, 3, figsize=(15, 5))

sns.distplot(df_test.loc[df_test["sentiment"]=='neutral', 'text_size'].rename("neutral"), label='text', ax=ax[0])

ax[0].legend()

sns.distplot(df_test.loc[df_test["sentiment"]=='positive', 'text_size'].rename("positive"), label='text', ax=ax[1])

ax[1].legend()

sns.distplot(df_test.loc[df_test["sentiment"]=='negative', 'text_size'].rename("negative"), label='text', ax=ax[2])

ax[2].legend()

plt.show()
fig, ax = plt.subplots(1, 3, figsize=(15, 5))



sns.distplot(df_train.loc[df_train["sentiment"]=='neutral', 'text_size'], label='neutral', ax=ax[0])

sns.distplot(df_train.loc[df_train["sentiment"]=='positive', 'text_size'], label='positive', ax=ax[0])

sns.distplot(df_train.loc[df_train["sentiment"]=='negative', 'text_size'], label='negative', ax=ax[0])

ax[0].legend()



sns.distplot(df_train.loc[df_train["sentiment"]=='neutral', 'selected_text_size'], label='neutral', ax=ax[1])

sns.distplot(df_train.loc[df_train["sentiment"]=='positive', 'selected_text_size'], label='positive', ax=ax[1])

sns.distplot(df_train.loc[df_train["sentiment"]=='negative', 'selected_text_size'], label='negative', ax=ax[1])

ax[1].legend()



sns.distplot(df_train.loc[df_train["sentiment"]=='neutral', 'text_selected_ratio'], label='neutral', ax=ax[2])

sns.distplot(df_train.loc[df_train["sentiment"]=='positive', 'text_selected_ratio'], label='positive', ax=ax[2])

sns.distplot(df_train.loc[df_train["sentiment"]=='negative', 'text_selected_ratio'], label='negative', ax=ax[2])

ax[2].legend()



plt.show()
print(f'Total count of empyt text : {sum(df_train["text_size"] == 0)}')

print(f'Total count of empyt selected_text : {sum(df_train["selected_text_size"] == 0)}')
from wordcloud import WordCloud

import nltk

from nltk.corpus import stopwords
# https://www.kaggle.com/ekhtiar/unintended-eda-with-tutorial-notes

def generate_word_cloud(df_data, text_col):

    # convert stop words to sets as required by the wordcloud library

    stop_words = set(stopwords.words("english"))

    

    data_neutral = " ".join(df_data.loc[df_data["sentiment"]=="neutral", text_col].map(lambda x: str(x).lower()))

    data_positive = " ".join(df_data.loc[df_data["sentiment"]=="positive", text_col].map(lambda x: str(x).lower()))

    data_negative = " ".join(df_data.loc[df_data["sentiment"]=="negative", text_col].map(lambda x: str(x).lower()))



    wc_neutral = WordCloud(max_font_size=100, max_words=100, background_color="white", stopwords=stop_words).generate(data_neutral)

    wc_positive = WordCloud(max_font_size=100, max_words=100, background_color="white", stopwords=stop_words).generate(data_positive)

    wc_negative = WordCloud(max_font_size=100, max_words=100, background_color="white", stopwords=stop_words).generate(data_negative)



    # draw the two wordclouds side by side using subplot

    fig, ax = plt.subplots(1, 3, figsize=(20, 20))

    ax[0].set_title("Neutral Wordcloud" , fontsize=10)

    ax[0].imshow(wc_neutral, interpolation="bilinear")

    ax[0].axis("off")

    

    ax[1].set_title("Positive Wordcloud", fontsize=10)

    ax[1].imshow(wc_positive, interpolation="bilinear")

    ax[1].axis("off")

    

    ax[2].set_title("Negative Wordcloud", fontsize=10)

    ax[2].imshow(wc_negative, interpolation="bilinear")

    ax[2].axis("off")

    plt.show()

    

    return [wc_neutral, wc_positive, wc_negative]
train_text_wc = generate_word_cloud(df_train, "text")
train_sel_text_wc = generate_word_cloud(df_train, "selected_text")
train_text_wc[1].words_
train_sel_text_wc[1].words_
train_text_wc[2].words_
train_sel_text_wc[2].words_
test_text_wc = generate_word_cloud(df_test, "text")
test_text_wc[1].words_
test_text_wc[2].words_