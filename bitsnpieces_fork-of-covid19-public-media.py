# imports

import numpy as np

import matplotlib.pyplot as plt

import json

import requests

import io

import gc

import re



import logging



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt

from wordcloud import WordCloud



# from tqdm import tqdm

from tqdm.notebook import tqdm





# pandas settings

pd.set_option('display.max_colwidth', -1)

pd.set_option('display.max_rows', 1000)

plt.rcParams['figure.figsize'] = [12, 8]





from nltk import download

download('stopwords')  # Download stopwords list.

from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))

from nltk import word_tokenize

download('punkt')  # Download data for tokenizer.



plt.rcParams['figure.figsize'] = [12, 8]
MAX_LEN = 3000   # 3000 chars



research = pd.read_csv('/kaggle/input/CORD-19-research-challenge/metadata.csv')

research['title_abstract'] = [str(research.loc[i,'title']) + ' ' + str(research.loc[i,'abstract']) for i in research.index ]

research['source'] = 'research'

research



news = pd.read_csv('/kaggle/input/covid19-public-media-dataset/covid19_articles.csv')

del news["Unnamed: 0"]

news['source'] = 'news'

news['title_abstract'] = [ news.loc[i,'title'] + '. ' + news.loc[i,'content'][:(MAX_LEN-len(news.loc[i,'title']))] for i in news.index  ]

news



data = pd.concat([research[['title_abstract','source', 'url']], news[['title_abstract', 'source', 'url']]]).rename(columns={'title_abstract':'title'}).drop_duplicates().reset_index(drop=True)



print('News:',news.shape)

print('Research:',research.shape)

print('Combined data:',data.shape)



del research

del news

gc.collect()



data
# Gensim word embeddings

# https://www.kaggle.com/raymishra/sentence-similarity-match

# https://radimrehurek.com/gensim/models/fasttext.html

# https://radimrehurek.com/gensim/models/keyedvectors.html



import gensim

from gensim.models import Word2Vec

from gensim.utils import simple_preprocess



from gensim.models.keyedvectors import KeyedVectors



filepath = "../input/gnewsvector/GoogleNews-vectors-negative300.bin"





from gensim.models import KeyedVectors

wv_from_bin = KeyedVectors.load_word2vec_format(filepath, binary=True) 



#extracting words7 vectors from google news vector

embeddings_index = {}

for word, vector in zip(wv_from_bin.vocab, wv_from_bin.vectors):

    coefs = np.asarray(vector, dtype='float32')

    embeddings_index[word] = coefs
# helper functions



def preprocess(doc):

#     doc = re.sub(r'[\W\d]+',' ',doc)  # Remove numbers and punctuation.

    doc = doc.lower()  # Lower the text.

    doc = word_tokenize(doc)  # Split into words.

    doc = [w for w in doc if not w in stop_words]  # Remove stopwords.

    doc = [w for w in doc if w.isalpha()]  # Remove numbers and punctuation.

    return doc



def avg_feature_vector(sentence, model, num_features):

#     words = sentence.lower().split()

#     words = preprocess(sentence)

    words = simple_preprocess(sentence)

    #feature vector is initialized as an empty array

    feature_vec = np.zeros((num_features, ), dtype='float32')

    n_words = 0

    for word in words:

        if word in embeddings_index.keys():

            n_words += 1

            feature_vec = np.add(feature_vec, model[word])

    if (n_words > 0):

        feature_vec = np.divide(feature_vec, n_words)

    return feature_vec



from scipy.spatial import distance

def calc_dist_cosine(s1, target, max_dist=0.5):

    ret = []

    for t in tqdm(target):

        tv = avg_feature_vector(t,model= embeddings_index, num_features=300)

        qv = avg_feature_vector(q,model= embeddings_index, num_features=300)

        dist = distance.cosine(tv, qv)

        if dist <= max_dist:

            ret.append([dist, t])

    df = pd.DataFrame(ret,columns=['dist','title']).reset_index(drop=True)

    return pd.merge(df, data, on='title', how='left').sort_values(by='dist', ascending=True).reset_index(drop=True)





# wv_from_bin.init_sims(replace=True)  # Normalizes the vectors in the word2vec class before calculating wmdistance

def calc_dist_wm(s1, target, max_dist=5.0):

    """

    Word mover distance. Slower than cosine similarity.

    https://markroxor.github.io/gensim/static/notebooks/WMD_tutorial.html

    """

    ret = []

    for t in tqdm(target):

#         print(t)

        dist = wv_from_bin.wmdistance(preprocess(s1), preprocess(t))

        if dist <= max_dist:

            ret.append([dist, t])

    df = pd.DataFrame(ret,columns=['dist','title']).reset_index(drop=True)

    return pd.merge(df, data, on='title', how='left').sort_values(by='dist', ascending=True).reset_index(drop=True)

       

def calc_dist(s1, target):

    """

    Dist interface

    """

    return calc_dist_cosine(s1, target)

#     return calc_dist_wm(s1, target)



# usage

# s1_afv = avg_feature_vector('Why the second proforma does not coincide with the first, what has changed', model= embeddings_index, num_features=300 )

# s2_afv = avg_feature_vector('Again came the proforma double.In the morning there was already a proforma with the same positions, but under a different number',model= embeddings_index, num_features=300)

# cos = distance.cosine(s1_afv, s2_afv)

# print(cos)

# calc_dist_wm('Why the second proforma does not coincide with the first, what has changed', ['Again came the proforma double.In the morning there was already a proforma with the same positions, but under a different number'])
# LDA

# https://www.kaggle.com/monsterspy/topic-modeling-with-lda

# https://www.machinelearningplus.com/nlp/topic-modeling-gensim-python/



from gensim.models import ldamodel

import gensim.corpora

from nltk import word_tokenize

from nltk.corpus import stopwords

stop = set(stopwords.words('english'))

stop.update(['href','br'])

from nltk.tokenize import RegexpTokenizer

tokenizer = RegexpTokenizer(r'\w+')



num_topics = 5



def train_lda(data_text):

    train_ = []

    for i in range(len(data_text)):

        train_.append([word for word in tokenizer.tokenize(data_text[i].lower()) if word not in stop])



    id2word = gensim.corpora.Dictionary(train_)

    corpus = [id2word.doc2bow(text) for text in train_]    # Term Document Frequency

    lda = ldamodel.LdaModel(corpus=corpus, id2word=id2word, num_topics=num_topics)

    return lda



def get_lda_topics(model, num_topics, topn=5):

    word_dict = {};

    for i in range(num_topics):

        words = model.show_topic(i, topn = topn);

        word_dict['Topic # ' + '{:02d}'.format(i+1)] = [i[0] for i in words];

    return pd.DataFrame(word_dict)

# lda = train_lda(data.title.values.tolist())

# lda_all_titles = get_lda_topics(lda, num_topics)

# lda_all_titles
def make_clickable(link):

    # target _blank to open new window

    return f'<a target="_blank" href="{link}">{link}</a>'

# df.style.format({'url': make_clickable})
def search(q, out_prefix="result"):

    res = calc_dist(q, data.title)

    res.to_csv(f'result_{out_prefix}.csv', index=False)



    # second iteration using word distance

#     res2 = calc_dist_wm(q, res.title)

#     res2.to_csv(f'result_{out_prefix}_wmd.csv', index=False)



#     lda = train_lda(res.title.values.tolist())

#     lda_res = get_lda_topics(lda, num_topics)

#     print(lda_res)

    

    topn = 20

    wc = WordCloud(background_color='white', stopwords=stop_words).generate(' '.join(res.title.values.tolist()[:topn]).lower())

    plt.imshow(wc)

    plt.axis('off')



    return res

q='ethical and social science considerations'

res = search(q, 'ethics_considerations')

res.head(20)[['title','url']].style.format({'url': make_clickable})
q='sustained education, access, and capacity building in the area of ethics'

res = search(q, 'sustained_ethics')

res.head(20)[['title','url']].style.format({'url': make_clickable})
q='qualitative assessment frameworks and secondary impacts of public health measures for prevention and control'

res = search(q, 'assessment')

res.head(20)[['title','url']].style.format({'url': make_clickable})
q='burden of responding to the outbreak and implementing public health measures affects the physical and psychological health of those providing care for Covid-19 patients'

res = search(q, 'burden')

res.head(20)[['title','url']].style.format({'url': make_clickable})
q='underlying drivers of fear, anxiety and stigma that fuel misinformation and rumor, particularly through social media'

res = search(q, 'misinformation')

res.head(20)[['title','url']].style.format({'url': make_clickable})