from lxml import etree

import pandas as pd

import numpy as np
patents = etree.parse("../input/ad20161018.xml")

root = patents.getroot()

assignments = list(list(root)[2])



def serialize(assn):

    srs = pd.Series()

    # Metadata

    srs['last-update-date'] = assn.find("assignment-record").find("last-update-date").find("date").text

    srs['recorded-date'] = assn.find("assignment-record").find("recorded-date").find("date").text

    srs['patent-assignors'] = "|".join([assn.find("name").text for assn in assn.find("patent-assignors")])

    srs['patent-assignees'] = "|".join([assn.find("name").text for assn in assn.find("patent-assignees")])

    # WIP---below.

    try:

        srs['patent-numbers'] = "|".join(

            ["|".join([d.find("doc-number").text for d in p.findall("document-id")])\

             for p in assn.find("patent-properties").findall("patent-property")]

        )

    except AttributeError:

        pass

    try:

        srs['patent-kinds'] = "|".join(

            ["|".join([d.find("kind").text for d in p.findall("document-id")])\

             for p in assn.find("patent-properties").findall("patent-property")]

        )

    except AttributeError:

        pass

    try:

        srs['patent-dates'] = "|".join(

            ["|".join([d.find("date").text for d in p.findall("document-id")])\

             for p in assn.find("patent-properties").findall("patent-property")]

        )    

    except AttributeError:

        pass

    try:

        srs['patent-countries'] = "|".join(

            ["|".join([d.find("country").text for d in p.findall("document-id")])\

             for p in assn.find("patent-properties").findall("patent-property")]

        )    

    except AttributeError:

        pass



    try:

        srs['title'] = "|".join(

            [p.find("invention-title").text for p in assn.find("patent-properties").findall("patent-property")]

        )        

    except AttributeError:

        pass

    return srs



flattened = pd.concat([serialize(assn) for assn in assignments], axis=1).T
del patents

del root

del assignments
assignments = flattened
assignments.head()
import nltk

import gensim
titles = assignments['title']

title_tokens = [nltk.word_tokenize(title) for title in\

                    np.concatenate(titles.map(str).map(str.title).map(lambda s: s.split("|")))]
title_tokens = [title for title in title_tokens if len(title_tokens) > 0]
len(title_tokens)
title_tokens[:3]
stemmer = nltk.stem.PorterStemmer()

titles_stemmed = [[stemmer.stem(token) for token in tokens] for tokens in title_tokens]
titles_stemmed[:3]
pd.Series(np.concatenate(titles_stemmed)).value_counts()
from nltk.corpus import stopwords
nltk.download("stopwords")
english_stopwords = set([word.title() for word in stopwords.words("english")])
stemmed_title_words = [[word for word in title if word not in english_stopwords] for title in titles_stemmed]
stemmed_title_words[:3]
word_counts = pd.Series(np.concatenate(stemmed_title_words)).value_counts()

singular_words = set(word_counts[pd.Series(np.concatenate(stemmed_title_words)).value_counts() == 1].index)
stemmed_title_common_words = [[word for word in title if word not in singular_words] for title in stemmed_title_words]
stemmed_title_common_words[:3]
non_empty_indices = [i for i in range(len(stemmed_title_common_words)) if len(stemmed_title_common_words[i]) > 0]
non_empty_indices[5000]
stemmed_title_common_words_nonnull = np.asarray(stemmed_title_common_words)[non_empty_indices]
classifiable_titles = np.asarray(title_tokens)[non_empty_indices]
dictionary = gensim.corpora.Dictionary(stemmed_title_common_words_nonnull)
str(dictionary.token2id)[:1000]
corpus = [dictionary.doc2bow(text) for text in stemmed_title_common_words_nonnull]
stemmed_title_common_words_nonnull[0], corpus[0]
stemmed_title_common_words_nonnull[100], corpus[100]
from gensim.models import TfidfModel
tfidf = TfidfModel(corpus)
stemmed_title_common_words_nonnull[0], corpus[0], tfidf[corpus[0]]
from gensim.models import LsiModel
corpus_tfidf = tfidf[corpus]

lsi = LsiModel(tfidf[corpus], id2word=dictionary, num_topics=10)

corpus_lsi = lsi[corpus_tfidf]
lsi.print_topics(10)
for scores in corpus_lsi[:5]:

    print(scores)
classifications = [np.argmax(np.asarray(corpus_lsi[i])[:,1]) for i in range(len(stemmed_title_common_words_nonnull))]
topics = pd.DataFrame({'topic': classifications, 'title': classifiable_titles})
topics['topic'].value_counts()
from IPython.display import display
for i in range(10):

    print("Topic", i + 1)

    display(topics.query('topic == @i').head(5))