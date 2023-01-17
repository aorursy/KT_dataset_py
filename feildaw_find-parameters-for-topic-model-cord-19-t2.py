# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

#for dirname, _, filenames in os.walk('/kaggle/input'):

#    for filename in filenames:

#        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import gensim

import matplotlib

import pickle

import gc
def readarticle(filepath):

    paperdata = {"paper_id" : None, "title" : None, "abstract" : None}

    with open(filepath) as file:

        filedata = json.load(file)

        paperdata["paper_id"] = filedata["paper_id"]

        paperdata["title"] = filedata["metadata"]["title"]

                

        if "abstract" in filedata:

            abstract = []

            for paragraph in filedata["abstract"]:

                abstract.append(paragraph["text"])

            abstract = '\n'.join(abstract)

            paperdata["abstract"] = abstract

        else:

            paperdata["abstract"] = []

    return paperdata



def read_multiple(jsonfiles_pathnames):

    papers = {"paper_id" : [], "title" : [], "abstract" : []}

    for filepath in jsonfiles_pathnames:

        paperdata = readarticle(filepath)

        if len(paperdata["abstract"]) > 0: 

            papers["paper_id"].append(paperdata["paper_id"])

            papers["title"].append(paperdata["title"])

            papers["abstract"].append(paperdata["abstract"])

            #papers["body_text"].append(paperdata["body_text"])

            #print("not none")

        #else:

            #print("none")

    print(len(papers["paper_id"]))

    print(len(papers["title"]))

    print(len(papers["abstract"]))

    return papers



def make_bigram(tokenized_data, min_count = 5, threshold = 100):

    bigram_phrases = gensim.models.Phrases(tokenized_data, min_count = min_count, threshold = threshold)

    #after Phrases a Phraser is faster to access

    bigram = gensim.models.phrases.Phraser(bigram_phrases)

    gc.collect()

    return bigram



def make_trigram(tokenized_data, min_count = 5, threshold = 100):

    bigram_phrases = gensim.models.Phrases(tokenized_data, min_count = min_count, threshold = threshold)

    trigram_phrases = gensim.models.Phrases(bigram_phrases[tokenized_data], threshold = 100)

    #after Phrases a Phraser is faster to access

    trigram = gensim.models.phrases.Phraser(trigram_phrases)

    gc.collect()

    return trigram



def readjson_retbodytext(jsonfiles_pathnames):

    print("reading json files")

    documents = read_multiple(jsonfiles_pathnames)

    print("writing documents dictionary to output for use in another kernel")

    with open("documents_dict.pkl", 'wb') as f:

        pickle.dump(documents, f)

    print("done writing documents dict.  Format is paper_id, title, body_text")

    gc.collect()

    return documents["abstract"]

    

def open_tokenize(jsonfiles_pathnames):

    

    body_text = readjson_retbodytext(jsonfiles_pathnames)

    

    print("removing stopwords, steming, and tokenizing.  This is expensive")

    tokenized_documents = gensim.parsing.preprocessing.preprocess_documents(body_text)

    print("done preprocessing documents. now writing to output to be used in another documents")

    with open("tokenized_documents.pkl", 'wb') as f:

        pickle.dump(tokenized_documents, f)

    print("done writing file")

    

    gc.collect()

    return tokenized_documents
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

def topic_distribution(query_string, id2word_dict, lda_model):

    tokenized_query = gensim.parsing.preprocessing.preprocess_string(query_string)



    print("tokens in query: %i" % (len(tokenized_query)))

    print(tokenized_query)

    

    vectorized_query = id2word_dict.doc2bow(tokenized_query)

    

    return lda_model[vectorized_query]    #query topic vector (distribution)

    

def corpus_similarities_print3(query_topicvec, index, documents_dict):

    similarities = index[query_topicvec]

    ranked_indices = sorted(enumerate(similarities), key = lambda item: -item[1])

    #papers = {"paper_id" : [], "title" : [], "abstract" : []}

    

    document_pids = documents_dict["paper_id"]

    document_titles = documents_dict["title"]

    #document_abstracts = documents_dict["abstract"]

    

    print(ranked_indices[0][0])

    topdex = ranked_indices[0][0]

    second = ranked_indices[1][0]

    third = ranked_indices[2][0]

    

    print("\nTOP RESULT TITLE: %s" % (document_titles[topdex]))

    print("TOP RESULT PID: %s" % (document_pids[topdex]))

    print(second)

    print("\nSecond RESULT TITLE: %s" % (document_titles[second]))

    print("Second RESULT PID: %s" % (document_pids[second]))

    print(third)

    print("\nThird RESULT TITLE: %s" % (document_titles[third]))

    print("Third RESULT PID: %s" % (document_pids[third]))

    gc.collect()
coherence_results_path = "/kaggle/input/try-lda-parameters/coherence_dict.pkl"

with open(coherence_results_path, "rb") as f:

    coherence_results = pickle.load(f)



#including the defintion for this dictionary in the comments for our reference



#coherence_dict = {"topic_counts" : topic_counts,

#                  "coherence_1gram_512features" : coherence_1gram_512features,

#                  "coherence_1gram_1024features" : coherence_1gram_1024features,

#                  "coherence_2gram_256features" : coherence_2gram_256features, 

#                  "coherence_2gram_512features" : coherence_2gram_512features,

#                  "coherence_2gram_1024features" : coherence_2gram_1024features}



start = coherence_results["topic_counts"][0]

length = len(coherence_results["topic_counts"])

stop = length + start



print(start)

print(length)

print(stop)



x = range(start, stop, 1)

matplotlib.pyplot.plot(x, coherence_results["coherence_1gram_512features"], label = "1gram_512feats")

matplotlib.pyplot.plot(x, coherence_results["coherence_1gram_1024features"], label = "1gram_1024feats")

matplotlib.pyplot.plot(x, coherence_results["coherence_2gram_256features"], label = "2gram_256feats")

matplotlib.pyplot.plot(x, coherence_results["coherence_2gram_512features"], label = "2gram_512feats")

matplotlib.pyplot.plot(x, coherence_results["coherence_2gram_1024features"], label = "2gram_1024feats")



matplotlib.pyplot.xlabel("Number of topics")

matplotlib.pyplot.ylabel("Coherence score")



matplotlib.pyplot.title("Coherence values for for gram 1 and 2, and features 256, 512, and 1024")

matplotlib.pyplot.legend()

matplotlib.pyplot.show()
tokenized_path = "/kaggle/input/preprocess-cord19/tokenized_documents.pkl"

print("opening %s" % str(tokenized_path)) 

with open(tokenized_path, "rb") as f:

    tokenized_documents = pickle.load(f)

print("done opening tokenized documents.  Optimizing")



bigram_path = "/kaggle/input/preprocess-cord19/bigram_model.pkl"

print("opening %s" % str(bigram_path))

with open(bigram_path, "rb") as f:

    bigram_model = pickle.load(f)

print("creating bigram documents")

tokenized_document = [bigram_model[document] for document in tokenized_documents]

print("done retrieving documents. lets optimize")



n_feats = 512

n_topics = 18

id2word_dict = create_dictionary(tokenized_documents, n_feats = n_feats)

tfcorpus = corpus_tf(id2word_dict, tokenized_documents)

print("training lda model with %i topics" % (n_topics))

lda_model = gensim.models.ldamodel.LdaModel(corpus = tfcorpus, num_topics = n_topics, id2word = id2word_dict, per_word_topics = False)

coherence_model = gensim.models.CoherenceModel(model = lda_model, texts = tokenized_documents, dictionary = id2word_dict, coherence = "c_v")

coherence_score = coherence_model.get_coherence()

print("Achieved coherence of: %f" % (coherence_score))

#build index for document similarity.  the similarity method used will be cosine similarity

print("creating index")

index = gensim.similarities.MatrixSimilarity(lda_model[tfcorpus])

print("done creating index")



documents_path = "/kaggle/input/preprocess-cord19/documents_dict.pkl"

with open(documents_path, "rb") as f:

    documents_dict = pickle.load(f)

print("done opening raw documents")



query = "What do we know about COVID-19 risk factors?"

query_topicvec = topic_distribution(query, id2word_dict, lda_model)

corpus_similarities_print3(query_topicvec, index, documents_dict)
query = " What have we learned from epidemiological studies?"

query_topicvec = topic_distribution(query, id2word_dict, lda_model)

corpus_similarities_print3(query_topicvec, index, documents_dict)
query = "Data on potential risks factors. Smoking, pre-existing pulmonary disease"

query_topicvec = topic_distribution(query, id2word_dict, lda_model)

corpus_similarities_print3(query_topicvec, index, documents_dict)
query = "Data on potential risks factors. Co-infections (determine whether co-existing respiratory/viral infections make the virus more transmissible or virulent) and other co-morbidities"

query_topicvec = topic_distribution(query, id2word_dict, lda_model)

corpus_similarities_print3(query_topicvec, index, documents_dict)
query = "Data on potential risks factors. Neonates and pregnant women"

query_topicvec = topic_distribution(query, id2word_dict, lda_model)

corpus_similarities_print3(query_topicvec, index, documents_dict)
query = "Data on potential risks factors. Socio-economic and behavioral factors to understand the economic impact of the virus and whether there were differences."

query_topicvec = topic_distribution(query, id2word_dict, lda_model)

corpus_similarities_print3(query_topicvec, index, documents_dict)
query = "Transmission dynamics of the virus, including the basic reproductive number, incubation period, serial interval, modes of transmission and environmental factors"

query_topicvec = topic_distribution(query, id2word_dict, lda_model)

corpus_similarities_print3(query_topicvec, index, documents_dict)
query = "Severity of disease, including risk of fatality among symptomatic hospitalized patients, and high-risk patient groups"

query_topicvec = topic_distribution(query, id2word_dict, lda_model)

corpus_similarities_print3(query_topicvec, index, documents_dict)
query = "Susceptibility of populations"

query_topicvec = topic_distribution(query, id2word_dict, lda_model)

corpus_similarities_print3(query_topicvec, index, documents_dict)
query = "Public health mitigation measures that could be effective for control"

query_topicvec = topic_distribution(query, id2word_dict, lda_model)

corpus_similarities_print3(query_topicvec, index, documents_dict)