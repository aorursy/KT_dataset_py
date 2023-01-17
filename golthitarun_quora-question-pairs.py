#authors:

#Shiva Ganga

#Tharunn Golthi

#Abhinaya

#Susmitha

#importing libraries numpy,pandas,mathplotlib for extracting, modifying and visualizing the data



import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt



from subprocess import check_output



# Any results you write to the current directory are saved as output.
#loading input train file to train

train = pd.read_csv("../input/train.csv")

#loading iput test file to test

test = pd.read_csv("../input/test.csv")

#printing the top

train.head()
test.head()
train.info()
test.info()
train_duplicate_mean = train['is_duplicate'].mean()

print ("mean of train data is_duplicate column",train_duplicate_mean)
pt = train.groupby('is_duplicate')['id'].count()

pt.plot.bar()
# plotting data for number of questions vs number of occurences of the question 

question_id_1 = train['qid1'].tolist()

question_id_2 = train['qid2'].tolist()



question_id = pd.Series(question_id_1+question_id_2)

plt.figure(figsize=(15,6))

plt.hist(question_id.value_counts(), bins= 30)

plt.yscale('log', nonposy='clip')
#using nltk corpus for stopwords

from nltk.corpus import stopwords as st

#stopwords

stopwords_set = set(st.words("english"))



#returns total words in a sentence

def word_dict(sentence):

    question_words_dict = {}

    for word in sentence.lower().split():

        if word not in stopwords_set:

            question_words_dict[word] = 1

    return question_words_dict

#calculating feature common_word_percentage for each row

def common_words_percentage(entry):

    question_1_words = word_dict(str(entry['question1']))

    question_2_words = word_dict(str(entry['question2']))

     

    if len(question_1_words) == 0 or len(question_2_words) == 0:

        return 0

    shared_in_q1 = [word for word in question_1_words.keys() if word in question_2_words]

    feature_Ratio = ( 2*len(shared_in_q1) )/(len(question_1_words)+len(question_2_words))

    return feature_Ratio
#calculating tfidf weights 

def tfidf_weights(entry):

    question_1_words = word_dict(str(entry['question1']))

    question_2_words = word_dict(str(entry['question2']))

    if len(question_1_words) == 0 or len(question_2_words) == 0:

        return 0

    

    common_wts_1 = [weights.get(w, 0) for w in question_1_words.keys() if w in question_2_words]  

    common_wts_2 = [weights.get(w, 0) for w in question_2_words.keys() if w in question_2_words]

    common_wts = common_wts_1 + common_wts_2

    whole_wts = [weights.get(w, 0) for w in question_1_words] + [weights.get(w, 0) for w in question_2_words]

    

    feature_tfidf = np.sum(common_wts) / np.sum(whole_wts)

    return feature_tfidf


list_of_questions = (train['question1'].str.lower().astype('U').tolist() + train['question2'].str.lower().astype('U').tolist())

#calcutaing Tfifs feature using inbuilt libraries

from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(min_df = 50,max_features = 3000000,ngram_range = (1,10))

X = vectorizer.fit_transform(list_of_questions)

idf = vectorizer.idf_

weights = (dict(zip(vectorizer.get_feature_names(), idf)))
#feature train data frame

X_TrainData = pd.DataFrame()

#feature test data frame

X_TestData = pd.DataFrame()

# adding common_word_percent feature to train data

X_TrainData['common_word_percent'] = train.apply(common_words_percentage, axis=1, raw=True)

# adding feature_ifidf feature to train data

X_TrainData['feature_ifidf'] = train.apply(tfidf_weights, axis = 1, raw = True)

Y_TrainData = train['is_duplicate'].values

# adding common_word_percent feature to test data

X_TestData['common_word_percent'] = test.apply(common_words_percentage, axis = 1, raw = True)

# adding feature_ifidf feature to test data

X_TestData['feature_ifidf'] = test.apply(tfidf_weights, axis = 1, raw = True)
# calculating jacardian similarity

import nltk

def jaccard_similarity_coefficient(row):

    if (type(row['question1']) is str) and (type(row['question2']) is str):

        words_1 = row['question1'].lower().split()

        words_2 = row['question2'].lower().split()

    else:

        #tokeninzing using nltk

        words_1 = nltk.word_tokenize(str(row['question1']))

        words_2 = nltk.word_tokenize(str(row['question2']))

   

    joint_words = set(words_1).union(set(words_2))

    intersection_words = set(words_1).intersection(set(words_2))

    return len(intersection_words)/len(joint_words)
# removing NA values in tarainig data

train = train.fillna("")
# adding jaccard distance feature to train and test data 

X_TrainData['Jacard_Distance'] = train.apply(jaccard_similarity_coefficient, axis = 1, raw = True)

X_TestData['Jacard_Distance'] = test.apply(jaccard_similarity_coefficient, axis = 1, raw = True)


from sklearn.metrics.pairwise import cosine_similarity as cs

import re, math

from collections import Counter



WORD = re.compile(r'\w+')

# calculating the cosine similarity between two vectors

def _cosine_similarity(vector_1, vector_2):

    

    common_keys = set(vector_1.keys()) & set(vector_2.keys())

    array1 = [vector_1[x]**2 for x in vector_1.keys()]

    array2 = [vector_2[x]**2 for x in vector_2.keys()]

    

    if not (math.sqrt(sum(array1)) * math.sqrt(sum(array2))):

        return 0.0

    else:

        return (float(sum([vector_1[x] * vector_2[x] for x in common_keys]))) / (math.sqrt(sum(array1)) * math.sqrt(sum(array2)))

# making sentence to vector format

def sentence_transform(sentence):

     words = WORD.findall(sentence)

     return Counter(words)

#method used to find cosine similarity for each row of data frame

def cosine_sim(row):

    vector1 = sentence_transform(str(row['question1']))

    vector2 = sentence_transform(str(row['question2']))

    sim = _cosine_similarity(vector1,vector2)

    return sim



X_TrainData['cosine_sim'] = train.apply(cosine_sim,axis = 1,raw = True )

X_TestData['cosine_sim'] = test.apply(cosine_sim,axis = 1,raw = True )
import csv, math, random , sys, random

from nltk.corpus import wordnet as wn

from nltk.corpus import brown

import math

import nltk

import sys

from nltk.corpus import stopwords

import numpy as np

import re

from pandas import read_csv

from nltk.corpus import wordnet as wn

from nltk.corpus import brown

import math

import nltk

import sys







##################

ALPHA = 0.2

BETA = 0.45

ETA = 0.4

PHI = 0.2

DELTA = 0.85



brown_freqs = dict()

N = 0





######################### word similarity ##########################

def get_best_synset_pair(word_1, word_2):

    """ 

    Choose the pair with highest path similarity among all pairs. 

    Mimics pattern-seeking behavior of humans.

    """

    max_sim = -1.0

    synsets_1 = wn.synsets(word_1)

    synsets_2 = wn.synsets(word_2)

    if len(synsets_1) == 0 or len(synsets_2) == 0:

        return None, None

    else:

        max_sim = -1.0

        best_pair = None, None

        for synset_1 in synsets_1:

            for synset_2 in synsets_2:

                sim = wn.path_similarity(synset_1, synset_2)

                if sim != None and sim > max_sim:

                    max_sim = sim

                    best_pair = synset_1, synset_2

        return best_pair





def length_dist(synset_1, synset_2):

    """

    Return a measure of the length of the shortest path in the semantic 

    ontology (Wordnet in our case as well as the paper's) between two 

    synsets.

    """

    l_dist = sys.maxsize

    if synset_1 is None or synset_2 is None:

        return 0.0

    if synset_1 == synset_2:

        # if synset_1 and synset_2 are the same synset return 0

        l_dist = 0.0

    else:

        wset_1 = set([str(x.name()) for x in synset_1.lemmas()])

        wset_2 = set([str(x.name()) for x in synset_2.lemmas()])

        if len(wset_1.intersection(wset_2)) > 0:

            # if synset_1 != synset_2 but there is word overlap, return 1.0

            l_dist = 1.0

        else:

            # just compute the shortest path between the two

            l_dist = synset_1.shortest_path_distance(synset_2)

            if l_dist is None:

                l_dist = 0.0

    # normalize path length to the range [0,1]

    return math.exp(-ALPHA * l_dist)





def hierarchy_dist(synset_1, synset_2):

    """

    Return a measure of depth in the ontology to model the fact that 

    nodes closer to the root are broader and have less semantic similarity

    than nodes further away from the root.

    """

    h_dist = sys.maxsize

    if synset_1 is None or synset_2 is None:

        return h_dist

    if synset_1 == synset_2:

        # return the depth of one of synset_1 or synset_2

        h_dist = max([x[1] for x in synset_1.hypernym_distances()])

    else:

        # find the max depth of least common subsumer

        hypernyms_1 = {x[0]: x[1] for x in synset_1.hypernym_distances()}

        hypernyms_2 = {x[0]: x[1] for x in synset_2.hypernym_distances()}

        lcs_candidates = set(hypernyms_1.keys()).intersection(

            set(hypernyms_2.keys()))

        if len(lcs_candidates) > 0:

            lcs_dists = []

            for lcs_candidate in lcs_candidates:

                lcs_d1 = 0

                if lcs_candidate in hypernyms_1:

                    lcs_d1 = hypernyms_1[lcs_candidate]

                lcs_d2 = 0

                if lcs_candidate in hypernyms_2:

                    lcs_d2 = hypernyms_2[lcs_candidate]

                lcs_dists.append(max([lcs_d1, lcs_d2]))

            h_dist = max(lcs_dists)

        else:

            h_dist = 0

    return ((math.exp(BETA * h_dist) - math.exp(-BETA * h_dist)) /

            (math.exp(BETA * h_dist) + math.exp(-BETA * h_dist)))





def word_similarity(word_1, word_2):

    synset_pair = get_best_synset_pair(word_1, word_2)

    return (length_dist(synset_pair[0], synset_pair[1]) *

            hierarchy_dist(synset_pair[0], synset_pair[1]))





######################### sentence similarity ##########################



def most_similar_word(word, word_set):

    """

    Find the word in the joint word set that is most similar to the word

    passed in. We use the algorithm above to compute word similarity between

    the word and each word in the joint word set, and return the most similar

    word and the actual similarity value.

    """

    max_sim = -1.0

    sim_word = ""

    for ref_word in word_set:

        sim = word_similarity(word, ref_word)

        if sim > max_sim:

            max_sim = sim

            sim_word = ref_word

    return sim_word, max_sim





def info_content(lookup_word):

    """

    Uses the Brown corpus available in NLTK to calculate a Laplace

    smoothed frequency distribution of words, then uses this information

    to compute the information content of the lookup_word.

    """

    global N

    if N == 0:

        # poor man's lazy evaluation

        for sent in brown.sents():

            for word in sent:

                word = word.lower()

                if word not in brown_freqs:

                    brown_freqs[word] = 0

                brown_freqs[word] = brown_freqs[word] + 1

                N = N + 1

    lookup_word = lookup_word.lower()

    n = 0 if lookup_word not in brown_freqs else brown_freqs[lookup_word]

    return 1.0 - (math.log(n + 1) / math.log(N + 1))





def semantic_vector(words, joint_words, info_content_norm):

    """

    Computes the semantic vector of a sentence. The sentence is passed in as

    a collection of words. The size of the semantic vector is the same as the

    size of the joint word set. The elements are 1 if a word in the sentence

    already exists in the joint word set, or the similarity of the word to the

    most similar word in the joint word set if it doesn't. Both values are 

    further normalized by the word's (and similar word's) information content

    if info_content_norm is True.

    """

    sent_set = set(words)

    semvec = np.zeros(len(joint_words))

    i = 0

    for joint_word in joint_words:

        if joint_word in sent_set:

            # if word in union exists in the sentence, s(i) = 1 (unnormalized)

            semvec[i] = 1.0

            if info_content_norm:

                semvec[i] = semvec[i] * math.pow(info_content(joint_word), 2)

        else:

            # find the most similar word in the joint set and set the sim value

            sim_word, max_sim = most_similar_word(joint_word, sent_set)

            semvec[i] = PHI if max_sim > PHI else 0.0

            if info_content_norm:

                semvec[i] = semvec[i] * info_content(joint_word) * info_content(sim_word)

        i = i + 1

    return semvec





def semantic_similarity(row):

    """

    Computes the semantic similarity between two sentences as the cosine

    similarity between the semantic vectors computed for each sentence.

    """

    info_content_norm = True

    sentence_1 = row['question1']

    sentence_2 = row['question2']

    

    words_1 = nltk.word_tokenize(sentence_1)

    words_2 = nltk.word_tokenize(sentence_2)

    joint_words = set(words_1).union(set(words_2))

    vec_1 = semantic_vector(words_1, joint_words, info_content_norm)

    vec_2 = semantic_vector(words_2, joint_words, info_content_norm)

    return np.dot(vec_1, vec_2.T) / (np.linalg.norm(vec_1) * np.linalg.norm(vec_2))

#if we implement semantic similarity it will take hours and hours of processing time 

#so we are not including the follwing feature 

# we have included the whole results and description of this feature in the report

"""X_TrainData['semantic_sim'] = train.apply(semantic_similarity,axis = 1,raw = True )

X_TestData['semantic_sim'] = test.apply(semantic_similarity,axis = 1,raw = True )"""

X_TrainData
from sklearn.cross_validation import train_test_split

# train test split validation data 20% and test data 80%

X_TrainData, X_ValidData, Y_TrainData, Y_ValidData = train_test_split(X_TrainData, Y_TrainData, test_size=0.20, random_state=4242)
import xgboost as xgb



xg_TrainData = xgb.DMatrix(X_TrainData, label=Y_TrainData)

xg_ValidData = xgb.DMatrix(X_ValidData, label=Y_ValidData)



watchlist = [(xg_TrainData, 'train'), (xg_ValidData, 'valid')]

#training using XGBoost using evalustion metric as logloss

bst = xgb.train({'objective':'binary:logistic','eval_metric':'logloss','eta':0.02,'max_depth' :5}, xg_TrainData, 500, watchlist, early_stopping_rounds=50, verbose_eval=10)
X_TestData.info()


xg_TestData = xgb.DMatrix(X_TestData)

xg_ValidData = xgb.DMatrix(X_ValidData)

#predited values using XG boost

Predict_TestData = bst.predict(xg_TestData)

Predict_ValidData = bst.predict(xg_ValidData)

#Roc metric

from sklearn.metrics import precision_recall_curve, auc, roc_curve

fpr, tpr, _ = roc_curve(Y_ValidData, Predict_ValidData)

roc_area = auc(fpr, tpr)

plt.plot(fpr, tpr)

np.round(roc_area, 10)
# precision Recall curve

precison, recall, _ = precision_recall_curve(Y_ValidData, Predict_ValidData)

plt.figure(figsize=(10,5))



plt.plot(recall, precison)

plt.xlabel('Recall')

plt.ylabel('Precision')

auc(recall, precison)
#final classes to result.csv

result = pd.DataFrame()

result['test_id'] = test['test_id']

result['is_duplicate'] = Predict_TestData

result.to_csv('result.csv', index=False)
