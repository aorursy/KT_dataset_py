# pip install pyldavis
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
#plotting tools
import pyLDAvis.gensim

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# data = pd.read_csv('/kaggle/input/million-headlines/abcnews-date-text.csv', error_bad_lines=False);
# data_text = data[['headline_text']]
# data_text['index'] = data_text.index
# documents = data_text
# documents = documents[:100]
documents = pd.DataFrame(columns=["headline_text","index"])
# documents = documents[:20]
documents["headline_text"]= ["I like to eat broccoli and bananas.",
                            "I ate a banana and spinach smoothie for breakfast.",
                            "Chinchillas and kittens are cute.",
                            "My sister adopted a kitten yesterday.",
                            "Look at this cute hamster munching on a piece of broccoli."]

documents["index"] = [i for i in range(5)]
documents.head()
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk import PorterStemmer
import numpy as np
np.random.seed(2018)
import nltk
def lemmatize_stemming(text):
#     return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))
    return  PorterStemmer().stem(text)
def preprocess(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            result.append(lemmatize_stemming(token))
    return result
doc_sample = documents[documents['index'] == 2].values[0][0]
print('original document: ')
words = []
for word in doc_sample.split(' '):
    words.append(word)
print(words)
print('\n\n tokenized and lemmatized document: ')
print(preprocess(doc_sample))
processed_docs = documents['headline_text'].map(preprocess)
processed_docs
dictionary = gensim.corpora.Dictionary(processed_docs)
count = 0
for k, v in dictionary.iteritems():
    print(k, v)
    count += 1
    if count > 10:
        break
# dictionary.filter_extremes(no_below=15, no_above=0.5, keep_n=100000)
bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]
bow_corpus
for k,v in dictionary.id2token.items():
    print(k,",",v)
freq_words = {i:0 for i in range(16) }
no_of_words = 0
for idxs in bow_corpus:
    for pair in idxs:
        freq_words[pair[0]] = freq_words[pair[0]] + pair[1] 
        no_of_words +=1
        
    
for k,v in freq_words.items():
    print(k,v)
word_prob = { k: v/no_of_words for k,v in freq_words.items()}
word_prob = np.array([ v/no_of_words for k,v in freq_words.items()])
word_prob
from gensim import corpora, models
tfidf = models.TfidfModel(bow_corpus)
corpus_tfidf = tfidf[bow_corpus]
from pprint import pprint
for doc in corpus_tfidf:
    pprint(doc)
    break
# lda_model = gensim.models.LdaMulticore(bow_corpus, num_topics=10, id2word=dictionary, passes=2, workers=2)

# build LDA model
lda_model = gensim.models.ldamodel.LdaModel(corpus=bow_corpus, id2word=dictionary, num_topics=2, random_state=42, 
                                            update_every=1, chunksize=100, passes=50, alpha='auto', 
                                            per_word_topics=True, dtype=np.float64,minimum_phi_value=0.01)
for idx, topic in lda_model.print_topics(-1):
    print('Topic: {} \nWords: {}'.format(idx, topic))
# gather topics and top words
top_topics = lda_model.top_topics(bow_corpus, topn=20)

# print results in an attractive format
pprint(top_topics)
fnames_argsort = np.asarray(list(dictionary.token2id.values()), dtype=np.int_)
topic = lda_model.state.get_lambda()
topic = topic / topic.sum(axis=1)[:, None]
topic_term_dists = topic[:, fnames_argsort]
topic_term_dists 
topic_given_term = topic_term_dists / topic_term_dists.sum()
topic_given_term
kernel = (topic_given_term * np.log((topic_given_term.T / 0.5).T))
distinctiveness = kernel.sum()
saliency = word_prob * distinctiveness
saliency
for i in top_topics:
    s = sum([k[0] for k in i[0]])
    print(s)
# Visualize the topics
vis = pyLDAvis.gensim.prepare(lda_model, bow_corpus, dictionary, sort_topics = False)


pyLDAvis.save_html(vis, 'LDA_Visualization.html')