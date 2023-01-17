import re

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from wordcloud import WordCloud, STOPWORDS

from copy import copy

import gensim

from gensim.utils import simple_preprocess

from gensim.parsing.preprocessing import STOPWORDS

from nltk.stem import WordNetLemmatizer, SnowballStemmer

from nltk.stem.porter import *

import nltk

from gensim import corpora, models

import pyLDAvis

import pyLDAvis.gensim

from keras.preprocessing.text import Tokenizer



pyLDAvis.enable_notebook()

np.random.seed(2018)

import warnings

warnings.filterwarnings('ignore')
with open("/kaggle/input/usa-presidential-debate-2020/presidential_debate_transcript.txt", "r") as file:

    text_data = file.read()
print(f"Length of transcript text: {len(text_data)}")
print(f"Text transcript excerpt:\n========================\n\n{text_data[:256]}")
regex_list = ["BIDEN:", "TRUMP:", "WALLACE:", "CROSSTALK"]

group = 0

texts = text_data.split("\n")

print(f"paragraphs: {len(texts)}")

for regex in regex_list:

    count = 0

    for text in texts:

        regex_pattern = re.compile(regex, re.UNICODE)

        results = [match.group(group) for match in regex_pattern.finditer(text)]

        count = count + len(results)

    print(f"speaker: {regex.rstrip(':')}, count: {count}")

stopwords = set(STOPWORDS)



def show_wordcloud(data, title = None):

    wordcloud = WordCloud(

        background_color='white',

        stopwords=stopwords,

        max_words=50,

        max_font_size=40, 

        scale=5,

        random_state=1

    ).generate(str(data))



    fig = plt.figure(1, figsize=(10,10))

    plt.axis('off')

    if title: 

        fig.suptitle(title, fontsize=20)

        fig.subplots_adjust(top=2.3)



    plt.imshow(wordcloud)

    plt.show()
show_wordcloud(text_data, "Prevalent words in the transcript")
filter_list = ["TRUMP", "BIDEN", "WALLACE", "CROSSTALK BIDEN", "CROSSTALK WALLACE", "CROSSTALK TRUMP", "CROSSTALK",

               "Vice President", "President Trump","President Biden", "Mr President", "Mr  President", "president"]



text_data_copy = copy(text_data)

for filter_item in filter_list:

    filtrate = re.compile(filter_item)

    text_data_copy = filtrate.sub(r"", text_data_copy)
show_wordcloud(text_data_copy, "Prevalent words in the transcript (filter repetitive words)")
def preprocess(text):

    result = []

    for token in gensim.utils.simple_preprocess(text):

        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 2:

            result.append(token)

    return result
text_sample = text_data[:256]

print('Original text: {}'.format(text_sample))

print('Tokenized text: {}'.format(preprocess(text_sample)))
texts = text_data.split("\n")

text_df = pd.DataFrame(texts)

text_df.columns = ["text"]

text_df.head()
preprocessed_text = text_df["text"].map(preprocess)
dictionary = gensim.corpora.Dictionary(preprocessed_text)

dictionary.filter_extremes(no_below=10, no_above=0.5, keep_n=75000)
bow_corpus = [dictionary.doc2bow(doc) for doc in preprocessed_text]

tfidf = models.TfidfModel(bow_corpus)

corpus_tfidf = tfidf[bow_corpus]
lda_model = gensim.models.LdaMulticore(corpus_tfidf, num_topics=20,

                                    id2word=dictionary, passes=2, workers=2)
topics = lda_model.print_topics(num_words=5)

for i, topic in enumerate(topics[:10]):

    print("Train topic {}: {}".format(i, topic))
bd5 = bow_corpus[5]

for i in range(len(bd5)):

    print("Word {} (\"{}\") appears {} time.".format(bd5[i][0], dictionary[bd5[i][0]],bd5[i][1]))
for index, score in sorted(lda_model[bd5], key=lambda tup: -1*tup[1]):

    print("\nScore: {}\t \nTopic: {}".format(score, lda_model.print_topic(index, 5)))
vis = pyLDAvis.gensim.prepare(lda_model, bow_corpus, dictionary)
pyLDAvis.save_html(vis, "LDAVis_text.html")