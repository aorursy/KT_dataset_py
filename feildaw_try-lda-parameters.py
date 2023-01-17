# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pickle

import gensim

import matplotlib

import gc
#no_n_below should be uint, ex: no_n_below = 3 or no_n_below = 5

#no_freq_above should be float [0,1], ex: no_freq_above = 0.5

#n_feats should be uint, ex: n_feats = 1024 or n_feats = 2048

def create_dictionary(tokenized_documents, n_feats, no_n_below = 3, no_freq_above = 0.5):

    print("creating dictionary")

    id2word_dict = gensim.corpora.Dictionary(tokenized_documents)

    print("done creating dictionary")

    

    print("prior dictionary len %i" % len(id2word_dict))

    id2word_dict.filter_extremes(no_below = no_n_below, no_above = no_freq_above, keep_n = n_feats, keep_tokens = None)

    print("current dictionary len %i" % len(id2word_dict))

    

    return id2word_dict   

    
def corpus_tf(id2word_dict, tokenized_documents):

    return [id2word_dict.doc2bow(document) for document in tokenized_documents]
def try_parameters(tokenized_documents, n_feats, n_topics):

    id2word_dict = create_dictionary(tokenized_documents, n_feats = n_feats)

    tfcorpus = corpus_tf(id2word_dict, tokenized_documents)

    print("training lda model with %i features and %i topics" % (n_feats, n_topics))

    lda_model = gensim.models.ldamodel.LdaModel(corpus = tfcorpus, num_topics = n_topics, id2word = id2word_dict, per_word_topics = False)

    coherence_model = gensim.models.CoherenceModel(model = lda_model, texts = tokenized_documents, dictionary = id2word_dict, coherence = "c_v")

    coherence_score = coherence_model.get_coherence()

    print("coherence for unknown ngram with %i features and %i topics: %f" % (n_feats, n_topics, coherence_score))

    gc.collect()

    return coherence_score
def loop_lda(tokenized_documents, 

                     tfcorpus, 

                     id2word_dict,

                     start, #suggest 2 or something

                     stop, # suggest 20 or similar

                     step,

                     per_word_topics = False): #compute list of topics for each word

    topic_counts = []

    coherence_scores = []

    for n_topics in range (start, stop, step):

        lda_model = gensim.models.ldamodel.LdaModel(corpus = tfcorpus, num_topics = n_topics, id2word = id2word_dict, per_word_topics = per_word_topics)

        coherence_model = gensim.models.CoherenceModel(model = lda_model, texts = tokenized_documents, dictionary = id2word_dict, coherence = "c_v")

        coherence_score = coherence_model.get_coherence()

        coherence_scores.append(coherence_score)

        topic_counts.append(n_topics)

        print("coherence of %f with %i topics" % (coherence_score, n_topics))

              

    return topic_counts, coherence_scores;

        

def loop_ntopics_lda(tokenized_documents, n_feats, start, stop, step):

    id2word_dict = create_dictionary(tokenized_documents, n_feats = n_feats)

    tfcorpus = corpus_tf(id2word_dict, tokenized_documents)

    topic_counts, coherence_scores = loop_lda(tokenized_documents, tfcorpus, id2word_dict, start, stop, step)

    gc.collect()

    return topic_counts, coherence_scores
ngram_bounds = (1,2)

n_feats_bounds = (512,2048)

n_topics_bounds = (1,20)



bounds = [ngram_bounds, n_feats_bounds, n_topics_bounds]



def lda_objective(X, tokenized_documents, tokenized_bigram_documents):

    ngram = int(round(X[0])) #bound should be [1,2]

    n_feats = int(round(X[1])) #bounds should be [512, 2048]

    n_topics = int(round(X[2])) #bouns should be [1,20]

    

    if ngram == 2:

        documents = tokenized_bigram_documents

        type_string = "tokenized_bigram_documents"

    else:

        documents = tokenized_documents

        type_string = "tokenized_documents"



    print("creating dictionary with %s for: %i %i %i" % (type_string, ngram, n_feats, n_topics))

    id2word_dict = create_dictionary(documents, n_feats = n_feats)



    print("done creating dictionary.  creating corpus for: %i %i %i" % (ngram, n_feats, n_topics))

    tfcorpus = corpus_tf(id2word_dict, documents)



    print("done creating corpus.  Building model for: %i %i %i" % (ngram, n_feats, n_topics))

    lda_model = gensim.models.ldamodel.LdaModel(corpus = tfcorpus, num_topics = n_topics, id2word = id2word_dict, per_word_topics = False)



    print("calculating coherence for: %i %i %i" % (ngram, n_feats, n_topics))

    coherence_model = gensim.models.CoherenceModel(model = lda_model, texts = documents, dictionary = id2word_dict, coherence = "c_v")

    coherence = coherence_model.get_coherence()

    #we want to MAX coherence.  but we will be using a 

    value2minimize = 1 - coherence

    return value2minimize
tokenized_path = "/kaggle/input/preprocess-cord19/tokenized_documents.pkl"

print("opening %s" % str(tokenized_path)) 

with open(tokenized_path, "rb") as f:

    tokenized_documents = pickle.load(f)

print("done opening tokenized documents.  Optimizing")



start = 2

stop = 20

step = 1



n_feats = 512

#coherence_1gram_512featurs_10topics = try_parameters(tokenized_documents, n_feats, n_topics)

topic_counts, coherence_1gram_512features = loop_ntopics_lda(tokenized_documents, n_feats, start, stop, step) #compute list of topics for each word

    

n_feats = 1024

topic_counts, coherence_1gram_1024features = loop_ntopics_lda(tokenized_documents, n_feats, start, stop, step) 



bigram_path = "/kaggle/input/preprocess-cord19/bigram_model.pkl"

print("opening %s" % str(bigram_path))

with open(bigram_path, "rb") as f:

    bigram_model = pickle.load(f)

print("creating bigram documents")

tokenized_document = [bigram_model[document] for document in tokenized_documents]

print("done retrieving documents. lets optimize")



gc.collect()



n_feats = 256

topic_counts, coherence_2gram_256features = loop_ntopics_lda(tokenized_documents, n_feats, start, stop, step) 



n_feats = 512

topic_counts, coherence_2gram_512features = loop_ntopics_lda(tokenized_documents, n_feats, start, stop, step) 



n_feats = 1024

topic_counts, coherence_2gram_1024features = loop_ntopics_lda(tokenized_documents, n_feats, start, stop, step) 



coherence_dict = {"topic_counts" : topic_counts,

                  "coherence_1gram_512features" : coherence_1gram_512features,

                  "coherence_1gram_1024features" : coherence_1gram_1024features,

                  "coherence_2gram_256features" : coherence_2gram_256features, 

                  "coherence_2gram_512features" : coherence_2gram_512features,

                  "coherence_2gram_1024features" : coherence_2gram_1024features}



dict_path = "coherence_dict.pkl"

print("saving %s" % (dict_path))

with open(dict_path, 'wb') as f:

    pickle.dump(coherence_dict, f)



#x = range(start, stop, step)

#matplotlib.pyplot.plot(x, coherence_scores_1gram_512feats)

#matplotlib.pyplot.plot(x, coherence_scores_1gram_1024feats)

#matplotlib.pyplot.plot(x, coherence_scores_1gram_2048feats)

#matplotlib.pyplot.xlabel("Number of topics")

#matplotlib.pyplot.ylabel("Coherence score")

#matplotlib.pyplot.legend((coherence_scores_2gram_256feats, coherence_scores_2gram_512feats), ("256feats", "512feats"))

#matplotlib.pyplot.title("Coherence values for ngram = 2")

#matplotlib.pyplot.show()
