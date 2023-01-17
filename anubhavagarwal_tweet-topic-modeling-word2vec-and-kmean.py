# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input/tweet_data/tweet_data"))

print(os.listdir("../input/tweet_data/tweet_data/word_dict_ngram"))



# Any results you write to the current directory are saved as output.
# -*- coding: utf-8 -*-

"""

Created on Thu Jun 19 11:59:34 2019



@author: AnubhavA

"""



"""English Word Segmentation using unigram and bigram data in Python



Reference:

    https://github.com/grantjenks/python-wordsegment



Source of unigram and bigram files:

    http://norvig.com/ngrams/ under the names count_1w.txt and count_2w.txt

 

"""



import io

import math



dir_path = "../input/tweet_data/tweet_data/word_dict_ngram"



class Segmenter(object):



    ALPHABET = set('abcdefghijklmnopqrstuvwxyz0123456789')  

    WORDS_FILENAME = dir_path + "/words.txt"

    BIGRAMS_FILENAME = dir_path + "/bigrams.txt"

    UNIGRAMS_FILENAME = dir_path +"/unigrams.txt"

    TOTAL = 1024908267229.0 #is the total number of words in the corpus ##Natural Language Corpus Data: Beautiful Data

    LIMIT = 24

   

    

    def __init__(self):

        "Initialize the class variables"

        self.unigrams = {}

        self.bigrams = {}

        self.total = 0.0

        self.limit = 0

        self.words = []

    

    @staticmethod

    def parse(filename):

        "Read `filename` and parse tab-separated file of word and count pairs."

        with io.open(filename, encoding='utf-8') as reader:

            lines = (line.split('\t') for line in reader)

            return dict((word, float(number)) for word, number in lines)

        



    def load(self):

        "Load unigram and bigram counts from local disk storage."

        self.unigrams.update(self.parse(self.UNIGRAMS_FILENAME))

        self.bigrams.update(self.parse(self.BIGRAMS_FILENAME))

        self.total = self.TOTAL

        self.limit = self.LIMIT

        with io.open(self.WORDS_FILENAME, encoding='utf-8') as reader:

            text = reader.read()

            self.words.extend(text.splitlines())





    def score(self, word, previous=None):

        "Score each `word` in the context of `previous` word."

        unigrams = self.unigrams

        bigrams = self.bigrams

        total = self.total



        if previous is None:

            if word in unigrams:



                # Probability of the given word.

                return unigrams[word] / total



            # Penalize words not found in the unigrams according

            # to their length

            return 10.0 / (total * 10 ** len(word))



        bigram = '{0} {1}'.format(previous, word)



        if bigram in bigrams and previous in unigrams:



            # Conditional probability of the word given the previous

            # word. 

            return bigrams[bigram] / total / self.score(previous)



        # Fall back to using the unigram probability.

        return self.score(word)





    def isegment(self, text):

        "Return iterator of words that is the best segmenation of `text`."

        memo = dict()



        def search(text, previous='<s>'):

            "Return max of candidates matching `text` given `previous` word."

            if text == '':

                return 0.0, []



            def candidates():

                "Generator of (score, words) pairs for all divisions of text."

                for prefix, suffix in self.divide(text):

                    prefix_score = math.log10(self.score(prefix, previous))



                    pair = (suffix, prefix)

                    if pair not in memo:

                        memo[pair] = search(suffix, prefix)

                    suffix_score, suffix_words = memo[pair]



                    yield (prefix_score + suffix_score, [prefix] + suffix_words)



            return max(candidates())



        # Avoid recursion limit issues by dividing text into chunks, segmenting

        # those chunks and combining the results together.



        clean_text = self.clean(text)

        size = 250

        prefix = ''



        for offset in range(0, len(clean_text), size):

            chunk = clean_text[offset:(offset + size)]

            _, chunk_words = search(prefix + chunk)

            prefix = ''.join(chunk_words[-5:])

            del chunk_words[-5:]

            for word in chunk_words:

                yield word



        _, prefix_words = search(prefix)



        for word in prefix_words:

            yield word





    def segment(self, text):

        "Return list of words that is the best segmenation of input `text`."

        return list(self.isegment(text))





    def divide(self, text):

        "Yield `(prefix, suffix)` pairs from input `text`."

        for pos in range(1, min(len(text), self.limit) + 1):

            yield (text[:pos], text[pos:])





    @classmethod

    def clean(cls, text):

        "Return `text` lower-cased with non-alphanumeric characters removed."

        alphabet = cls.ALPHABET

        text_lower = text.lower()

        letters = (letter for letter in text_lower if letter in alphabet)

        return ''.join(letters)

##importing required libraries



import string

import re

from collections import defaultdict



##importing NLTK packages

import nltk

nltk.download('wordnet')

nltk.download('stopwords')

nltk.download('punkt')

from nltk.tokenize import word_tokenize

from nltk.corpus import stopwords

from nltk.stem.wordnet import WordNetLemmatizer



##importing Gensim packages

import gensim

import gensim.models.keyedvectors as word2vec

from gensim import corpora



##importing PyLDAvis for LDA result visualization

import pyLDAvis.gensim
stop = set(stopwords.words('english'))

words_nltk = set(nltk.corpus.words.words()) #english words corpus of 235892 words - nltk

punct = string.punctuation

exclude_punct = set(punct)

exclude_punct_no_hash = set(punct.replace('#','')) #puntuations without hash character

lemma = WordNetLemmatizer()


def extract_hash_tags(tweet):

    "extract hashtags from the tweet"

    return set(part[1:] for part in tweet.split() if part.startswith('#'))



def word_segment(joinedword):

    "extract segmented words from hashtags"

    clean = segmenter.clean(joinedword) 

    broken_hashtag = segmenter.isegment(clean)    

    return broken_hashtag





def remove_non_eng_words_and_alphanum(twt_list):

    "remove non-eng words and words having alphanumeric from list of tweets"

    clean_twt_list=[]

    for twt in twt_list:

        ##Removing non-English words and alphanumeric

        new_clean_twt = [w for w in twt if w in words_nltk and w.isalpha()] 

        if len(new_clean_twt)>0:

            clean_twt_list.append(new_clean_twt)

    return clean_twt_list
##importing gensim preprocessing for text cleaning

from gensim.parsing.preprocessing import preprocess_string, strip_tags, strip_punctuation, strip_multiple_whitespaces, strip_numeric, remove_stopwords, strip_short, stem_text



CUSTOM_FILTERS = [lambda x: x.lower(), strip_tags, strip_punctuation, strip_multiple_whitespaces,strip_numeric, 

                  remove_stopwords,strip_short]



def clean_text(tweet):

    "pre-process and clean a tweet by doing strip_tags, strip_punctuation, strip_multiple_whitespaces,strip_numeric, remove_stopwords, strip_short words"

    tweet = preprocess_string(tweet, CUSTOM_FILTERS)

    return tweet

## list of hashtags in tweet corpus 

hashtags = []



def clean(tweet):

    "pre-process and clean a tweet - ASCII_free, punc_free, short_word_free, gensim pre-processing, repeat_free"

    non_ASCII_free = (tweet.encode('ascii', 'ignore')).decode("utf-8")

    punc_free = ''.join([ch for ch in non_ASCII_free if ch not in exclude_punct_no_hash])

    short_word_free = ' '.join(word for word in punc_free.split())

    

    ##building list of hashtags in the tweet corpus

    hash_tags = extract_hash_tags(short_word_free)

    for tag in hash_tags:

        if(len(tag)>0):

            hashtags.append(tag.lower())

    



    

    ##----- below line of code is replaced by gensim pre-processing - clean_text() function -----

    #stop_free = ' '.join([i for i in short_word_free.lower().split() if i not in stop])    

    #normalized = ' '.join(lemma.lemmatize(word) for word in stop_free.split())

    #http_free = ' '.join(re.sub(r'http\S+', '', word) for word in normalized.split())     

    #hash_punc_free = ''.join([ch for ch in http_free if ch not in {'#'}])

  

    

    cleaned_tweet = clean_text(short_word_free)   

    repeat_free = list(set(cleaned_tweet))

                

    return repeat_free

twt_all = []

clean_twt_Clint_Eastwood = []

clean_twt_Bradley_Cooper = []

clean_twt_Chris_Kyle = []



name_Clint_Eastwood = ['clint', 'eastwood', 'clinteastwood', 'clint eastwood']

name_Bradley_Cooper = ['bradley', 'bradleycooper', 'bradley cooper']

name_Chris_Kyle = ['chris', 'kyle', 'chriskyle', 'chris kyle']





twt_file = open("../input/tweet_data/tweet_data/tweets.txt", "r", encoding="utf8")

print("loaded data from tweet corpus file")

for twt in twt_file: 

    twt_all.append(twt)

print("Sample tweets:")

twt_all[0:5]
clean_twt_all = [clean(twt) for twt in twt_all]

print('Total tweets count: {0}'.format(len(clean_twt_all)))



##unique hashtag list

unique_hashtags = list(set(hashtags))

hashtags =  unique_hashtags



print("Total {0} unique hashtags found! ".format(len(hashtags)))

#Total 5750 unique hashtags found! 



print("Few sample hashtags:")

print(list(hashtags)[:10])



## create a hash tag dictionery, key=hash_tag, value=segmented words

hash_tag_dict={}

segmenter = Segmenter()  

load = segmenter.load()





for hashtag in hashtags:

    clean = segmenter.clean(hashtag) 

    isegment = segmenter.isegment(clean) 

    word_segment= list(isegment)    

    output = ' '.join(word for word in word_segment)

    hash_tag_dict.update( {hashtag : output} )

    

print("created hash tag dictionery with hashtag and segmented words")
## replace all hashtags with segmented words in clean tweet corpus

for twt in clean_twt_all:

    for index, word in enumerate(twt):

        if word in hash_tag_dict.keys():

            twt[index] = hash_tag_dict[word]   

    
clean_twt_Clint_Eastwood = [twt for twt in clean_twt_all if any(name in twt for name in name_Clint_Eastwood)]

clean_twt_Bradley_Cooper = [twt for twt in clean_twt_all if any(name in twt for name in name_Bradley_Cooper)]

clean_twt_Chris_Kyle = [twt for twt in clean_twt_all if any(name in twt for name in name_Chris_Kyle)]



print("Count of Tweets for Clint Eastwood: {0}".format(len(clean_twt_Clint_Eastwood)))

print("Count of Tweets for Bradley Cooper: {0}".format(len(clean_twt_Bradley_Cooper)))

print("Count of Tweets for Chris Kyle: {0}".format(len(clean_twt_Chris_Kyle)))



## remove non-eng words and alphanum, 

clean_twt_Clint_Eastwood = remove_non_eng_words_and_alphanum(clean_twt_Clint_Eastwood)

print("sample tweet for Clint_Eastwood: {0}".format(clean_twt_Clint_Eastwood[1]))



## remove non-eng words and alphanum,

clean_twt_Bradley_Cooper = remove_non_eng_words_and_alphanum(clean_twt_Bradley_Cooper)

print("sample tweet for Bradley_Cooper: {0}".format(clean_twt_Bradley_Cooper[1]))
## remove non-eng words and alphanum, 

clean_twt_Chris_Kyle = remove_non_eng_words_and_alphanum(clean_twt_Chris_Kyle)

print("sample tweet for Chris_Kyle: {0}".format(clean_twt_Chris_Kyle[3]))
def display_topics(model, feature_names, no_top_words):

    topic_dict = {}

    for topic_idx, topic in enumerate(model.components_):

        topic_dict["Topic %d words" % (topic_idx)]= ['{}'.format(feature_names[i])

                        for i in topic.argsort()[:-no_top_words - 1:-1]]

        topic_dict["Topic %d weights" % (topic_idx)]= ['{:.1f}'.format(topic[i])

                        for i in topic.argsort()[:-no_top_words - 1:-1]]

    return pd.DataFrame(topic_dict)
## creating Latent Dirichlet Allocation model for tweets of Clint_Eastwood

dictionary = corpora.Dictionary(clean_twt_Clint_Eastwood)

doc_term_matrix = [dictionary.doc2bow(twt) for twt in clean_twt_Clint_Eastwood]

Lda = gensim.models.ldamodel.LdaModel

ldamodel_CE = Lda(doc_term_matrix, num_topics = 5, id2word = dictionary, passes=50, random_state=100,

                                           update_every=1, chunksize=10,per_word_topics=True)

print("Topics for Clint_Eastwood")

#print(ldamodel_CE.print_topics(num_topics=5, num_words=10))



for index, topic in ldamodel_CE.show_topics(formatted=False, num_words= 10):

    print('Topic: {} \nWords: {}'.format(index + 1, [w[0] for w in topic]))



## creating Latent Dirichlet Allocation model for tweets of Bradley_Cooper



dictionary = corpora.Dictionary(clean_twt_Bradley_Cooper)

doc_term_matrix = [dictionary.doc2bow(twt) for twt in clean_twt_Bradley_Cooper]

Lda = gensim.models.ldamodel.LdaModel

ldamodel_BC = Lda(doc_term_matrix, num_topics = 5, id2word = dictionary, passes=50)

print("Topics for Bradley_Cooper")

#print(ldamodel_BC.print_topics(num_topics=5, num_words=10))

for index, topic in ldamodel_BC.show_topics(formatted=False, num_words= 10):

    print('Topic: {} \nWords: {}'.format(index + 1, [w[0] for w in topic]))

## creating Latent Dirichlet Allocation model for tweets of Chris_Kyle



dictionary = corpora.Dictionary(clean_twt_Chris_Kyle)

doc_term_matrix = [dictionary.doc2bow(twt) for twt in clean_twt_Chris_Kyle]

Lda = gensim.models.ldamodel.LdaModel

ldamodel_CK = Lda(doc_term_matrix, num_topics = 5, id2word = dictionary, passes=50)

print("Topics for Chris_Kyle")

#print(ldamodel_CK.print_topics(num_topics=5, num_words=10))

for index, topic in ldamodel_CK.show_topics(formatted=False, num_words= 10):

    print('Topic: {} \nWords: {}'.format(index + 1, [w[0] for w in topic]))

## running this is breaking the kernel, if require to visualize please uncomment



#import pyLDAvis.gensim

#pyLDAvis.enable_notebook()



#print("LDA model visualization for Clint_Eastwood")

#vis_CE = pyLDAvis.gensim.prepare(ldamodel_CE, doc_term_matrix, dictionary=ldamodel_CE.id2word)

#vis_CE

#print("LDA model visualization for Chris_Kyle")

#vis_CK = pyLDAvis.gensim.prepare(ldamodel_CK, doc_term_matrix, dictionary=ldamodel_CK.id2word)

#vis_CK

#print("LDA model visualization for Bradley_Cooper")

#vis_BC = pyLDAvis.gensim.prepare(ldamodel_BC, doc_term_matrix, dictionary=ldamodel_BC.id2word)

#vis_BC

EMBEDDING_DIM = 100



# train word2vec model for twt_Clint_Eastwood

model_CE = gensim.models.Word2Vec(sentences=clean_twt_Clint_Eastwood, size=EMBEDDING_DIM, window=4, workers=4, min_count=1)

words = list(model_CE.wv.vocab)

print('Vocabulary size for tweets of Clint_Eastwood: %d' % len(words))

# save model in ASCII (word2vec) format

filename = '/tmp/tweet_CE_embedding_word2vec.bin'

model_CE.wv.save(filename)

print('model saved to /tmp/tweet_CE_embedding_word2vec.bin')
# train word2vec model for Bradley_Cooper

model_BC = gensim.models.Word2Vec(sentences=clean_twt_Bradley_Cooper, size=EMBEDDING_DIM, window=4, workers=4, min_count=1)

words = list(model_BC.wv.vocab)

print('Vocabulary size for tweets of Bradley_Cooper: %d' % len(words))

# save model in ASCII (word2vec) format

filename = '/tmp/tweet_BC_embedding_word2vec.bin'

model_BC.wv.save(filename)

print('model saved to /tmp/tweet_BC_embedding_word2vec.bin')
# train word2vec model for twt_Chris_Kyle

model_CK = gensim.models.Word2Vec(sentences=clean_twt_Chris_Kyle, size=EMBEDDING_DIM, window=4, workers=4, min_count=1)

words = list(model_CK.wv.vocab)

print('Vocabulary size for tweets of Chris_Kyle: %d' % len(words))

# save model in ASCII (word2vec) format

filename = '/tmp/tweet_CK_embedding_word2vec.bin'

model_CK.wv.save(filename)

print('model saved to /tmp/tweet_CK_embedding_word2vec.bin')
#print(os.listdir("/tmp"))

#### uncomment to load the trained embedding model from disk that we have trained in above code ----------

##model_BC = word2vec.KeyedVectors.load('/tmp/tweet_BC_embedding_word2vec.bin')

##model_CE = word2vec.KeyedVectors.load('/tmp/tweet_CE_embedding_word2vec.bin')

##model_CK = word2vec.KeyedVectors.load('/tmp/tweet_CK_embedding_word2vec.bin')

### importing packages for K-Mean clustering and KDTree

from sklearn.cluster import KMeans

from sklearn.neighbors import KDTree

import pandas as pd



pd.set_option('display.max_columns', None)

pd.set_option('display.max_rows', None)



## Initalize a k-means object and use it to extract centroids

def clustering_on_wordvecs(word_vectors, num_clusters):

    kmeans_clustering = KMeans(n_clusters = num_clusters, init='k-means++')

    idx = kmeans_clustering.fit_predict(word_vectors)    

    return kmeans_clustering.cluster_centers_, idx





##Top K words on each cluster center.

def get_top_words(index2word, k, centers, wordvecs):

    tree = KDTree(wordvecs)

    

    #Closest points for each Cluster center is used to query the closest 20 points to it.

    closest_points = [tree.query(np.reshape(x, (1, -1)), k=k) for x in centers]

    closest_words_idxs = [x[1] for x in closest_points]



    #Word Index is queried for each position in the above array, and added to a Dictionary.

    closest_words = {}

    for i in range(0, len(closest_words_idxs)):

        closest_words['Cluster #' + str(i+1).zfill(2)] = [index2word[j] for j in closest_words_idxs[i][0]]



    #A DataFrame is generated from the dictionary.

    df = pd.DataFrame(closest_words)

    df.index = df.index+1



    return df;
Z = model_CE.wv.syn0;

print('shape of word vector for twt_Clint_Eastwood {0}:'.format(Z.shape))



centers, clusters = clustering_on_wordvecs(Z, 5)

centroid_map = dict(zip(model_CE.wv.index2word, clusters))



top_words_CE = get_top_words(model_CE.wv.index2word, 20, centers, Z)



print('top words around Clint_Eastwood:')

top_words_CE
Z = model_BC.wv.syn0;

print('shape of word vector for twt_Bradley_Cooper {0}:'.format(Z.shape))



centers, clusters = clustering_on_wordvecs(Z, 5)

centroid_map = dict(zip(model_BC.wv.index2word, clusters))



top_words_BC = get_top_words(model_BC.wv.index2word, 20, centers, Z)



print('top words around Bradley_Cooper:')

top_words_BC
Z = model_CK.wv.syn0;

print('shape of word vector for twt_Chris_Kyle {0}:'.format(Z.shape))



centers, clusters = clustering_on_wordvecs(Z, 5)

centroid_map = dict(zip(model_CK.wv.index2word, clusters))



top_words_CK = get_top_words(model_CK.wv.index2word, 20, centers, Z)



print('top words around Bradley_Cooper:')

top_words_CK