import numpy as np

import pandas as pd

import os

import seaborn as sns

from wordcloud import WordCloud, STOPWORDS

import matplotlib.pyplot as plt

from tqdm import tqdm_notebook

%matplotlib inline 

stopwords = set(STOPWORDS)
DATA_FOLDER = "/kaggle/input/tweet-sentiment-extraction/"

train_df = pd.read_csv(os.path.join(DATA_FOLDER, "train.csv"))

test_df = pd.read_csv(os.path.join(DATA_FOLDER, "test.csv"))
print(f"train: {train_df.shape}  test: {test_df.shape}")
train_df.head()
test_df.head()
def show_wordcloud(data, title = None):

    wordcloud = WordCloud(

        background_color='white',

        stopwords=stopwords,

        max_words=200,

        max_font_size=40, 

        scale=5,

        random_state=1

    ).generate(str(data))



    fig = plt.figure(1, figsize=(10,8))

    plt.axis('off')

    if title: 

        fig.suptitle(title, fontsize=14)

        fig.subplots_adjust(top=2.3)

    plt.imshow(wordcloud)

    plt.show()
show_wordcloud(train_df['text'], 'train: text')
show_wordcloud(train_df['selected_text'], 'train: selected text')
show_wordcloud(test_df['text'], 'test: text')
def plot_sentiment_count(data_df, title):

    plt.figure(figsize=(8,6))

    sns.countplot(data_df['sentiment'])

    plt.title(title)

    plt.show()
plot_sentiment_count(train_df, "Sentiment distribution: train")
plot_sentiment_count(test_df, "Sentiment distribution: test")
for sentiment in train_df.sentiment.unique():

    show_wordcloud(train_df.loc[train_df['sentiment']==sentiment, 'text'], f'train data - (sentiment: {sentiment}): text')
for sentiment in train_df.sentiment.unique():

    show_wordcloud(train_df.loc[train_df['sentiment']==sentiment, 'selected_text'], f'train data - (sentiment: {sentiment}): selected text')
for sentiment in test_df.sentiment.unique():

    show_wordcloud(test_df.loc[test_df['sentiment']==sentiment, 'text'], f'test data - (sentiment: {sentiment}): text')
punct_mapping = "/-'?!.,#$%\'()*+-/:;<=>@[\\]^_`{|}~" + '""“”’' + '∞θ÷α•à−β∅³π‘₹´°£€\×™√²—–&'

punct_mapping += '©^®` <→°€™› ♥←×§″′Â█½à…“★”–●â►−¢²¬░¶↑±¿▾═¦║―¥▓—‹─▒：¼⊕▼▪†■’▀¨▄♫☆é¯♦¤▲è¸¾Ã⋅‘∞∙）↓、│（»，♪╩╚³・╦╣╔╗▬❤ïØ¹≤‡√'



puncts = {"‘": "'", "´": "'", "°": "", "€": "e", "—": "-", "–": "-", "’": "'", "_": "-", "`": "'", '“': '"', '”': '"', '“': '"', "£": "e", '∞': 'infinity', 'θ': 'theta', '÷': '/', 'α': 'alpha', '•': '.', 'à': 'a', '−': '-', 'β': 'beta', '∅': '', '³': '3', 'π': 'pi', '…': ' '}



def clean_special_chars(text, punct, mapping):

    '''

    credits to: https://www.kaggle.com/christofhenkel/how-to-preprocessing-when-using-embeddings 

    credits to: https://www.kaggle.com/anebzt/quora-preprocessing-model

    input: current text, punctuations, punctuation mapping

    output: cleaned text

    '''

    for p in mapping:

        text = text.replace(p, mapping[p])

    for p in punct:

        text = text.replace(p, f' {p} ') 

    return text
train_df['text'] = train_df['text'].fillna("")

train_df['selected_text'] = train_df['selected_text'].fillna("")

test_df['text'] = test_df['text'].fillna("")
train_df['cleaned_text'] = train_df['text'].apply(lambda x: clean_special_chars(x, punct_mapping, puncts))

test_df['cleaned_text'] = test_df['text'].apply(lambda x: clean_special_chars(x, punct_mapping, puncts))

train_df['cleaned_selected_text'] = train_df['selected_text'].apply(lambda x: clean_special_chars(x, punct_mapping, puncts))
import gensim

def preprocess(text):

    result = []

    for token in gensim.utils.simple_preprocess(text):

        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 2:

            result.append(token)

    return result
train_df['preproc_text'] = train_df['cleaned_text'].apply(lambda x: preprocess(x))

test_df['preproc_text'] = test_df['cleaned_text'].apply(lambda x: preprocess(x))

train_df['preproc_selected_text'] = train_df['cleaned_selected_text'].apply(lambda x: preprocess(x))
train_df['cnt_text'] = train_df['preproc_text'].apply(lambda x: len(x))

test_df['cnt_text'] = test_df['preproc_text'].apply(lambda x: len(x))

train_df['cnt_selected_text'] = train_df['preproc_selected_text'].apply(lambda x: len(x))
def plot_feature_density(data_df, feature='cnt_text', title=''):

    plt.figure(figsize=(8,6))

    for sentiment in data_df.sentiment.unique():

        sns.distplot(data_df.loc[data_df['sentiment']==sentiment, feature], kde=True, hist=False, label=sentiment)

    plt.title(title)

    plt.show()
plot_feature_density(train_df, 'cnt_text', 'Word count distribution - text - train')
plot_feature_density(test_df, 'cnt_text', 'Word count distribution - text - test')
plot_feature_density(train_df, 'cnt_selected_text', 'Word count distribution - selected text - train')