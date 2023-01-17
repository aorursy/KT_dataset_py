# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.

# All Import Statements Defined Here

# Note: Do not add to this list.

# All the dependencies you need, can be installed by running .

# ----------------

import sys

assert sys.version_info[0]==3

assert sys.version_info[1] >= 5



import pprint

import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = [10, 5]

import nltk

nltk.download('reuters')

from nltk.corpus import reuters

import numpy as np

import random

import scipy as sp

from sklearn.decomposition import TruncatedSVD

from sklearn.decomposition import PCA



START_TOKEN = '<START>'

END_TOKEN = '<END>'



np.random.seed(0)

random.seed(0)

# ----------------



def read_corpus(category="crude"):

    """ Read files from the specified Reuter's category.

        Params:

            category (string): category name

        Return:

            list of lists, with words from each of the processed files

    """

    files = reuters.fileids(category)

    return [[START_TOKEN] + [w.lower() for w in list(reuters.words(f))] + [END_TOKEN] for f in files]



reuters_corpus = read_corpus()





# Answer to Question 1.1: Implement distinct_words [code] (2 points)

#===============================================================================================================



def distinct_words(corpus):

    """ Determine a list of distinct words for the corpus.

        Params:

            corpus (list of list of strings): corpus of documents

        Return:

            corpus_words (list of strings): list of distinct words across the corpus, sorted (using python 'sorted' function)

            num_corpus_words (integer): number of distinct words across the corpus

    """

    corpus_words = []

    num_corpus_words = -1



    # ------------------

    # Write your implementation here.

    # Flattening the corpus

    split_corpus = [y for word in corpus for y in word]

    # Adding words from list if not present in corpus_words(corpus_words is the output or unique word list)

    for word in split_corpus:

        if word not in corpus_words:

            corpus_words.append(word)

            # Print list

    corpus_words = sorted(corpus_words)

    num_corpus_words = len(corpus_words)

    # ------------------



    return corpus_words, num_corpus_words



# The Answer to Question 1.1 ends here

#===============================================================================================================



# ---------------------

# Run this sanity check

# Note that this not an exhaustive check for correctness.

# ---------------------



# Define toy corpus

test_corpus = ["START All that glitters isn't gold END".split(" "), "START All's well that ends well END".split(" ")]

test_corpus_words, num_corpus_words = distinct_words(test_corpus)



# Correct answers

ans_test_corpus_words = sorted(list(set(["START", "All", "ends", "that", "gold", "All's", "glitters", "isn't", "well", "END"])))

ans_num_corpus_words = len(ans_test_corpus_words)



# Test correct number of words

assert(num_corpus_words == ans_num_corpus_words), "Incorrect number of distinct words. Correct: {}. Yours: {}".format(ans_num_corpus_words, num_corpus_words)



# Test correct words

assert (test_corpus_words == ans_test_corpus_words), "Incorrect corpus_words.\nCorrect: {}\nYours:   {}".format(str(ans_test_corpus_words), str(test_corpus_words))



# Print Success

print ("-" * 80)

print("Passed All Tests!")

print ("-" * 80)



#=====================================================================================================================

def compute_co_occurrence_matrix(corpus, window_size=4):

    """ Compute co-occurrence matrix for the given corpus and window_size (default of 4).



        Note: Each word in a document should be at the center of a window. Words near edges will have a smaller

              number of co-occurring words.



              For example, if we take the document "START All that glitters is not gold END" with window size of 4,

              "All" will co-occur with "START", "that", "glitters", "is", and "not".



        Params:

            corpus (list of list of strings): corpus of documents

            window_size (int): size of context window

        Return:

            M (numpy matrix of shape (number of corpus words, number of corpus words)):

                Co-occurence matrix of word counts.

                The ordering of the words in the rows/columns should be the same as the ordering of the words given by the distinct_words function.

            word2Ind (dict): dictionary that maps word to index (i.e. row/column number) for matrix M.

    """

    words, num_words = distinct_words(corpus)

    M = None

    word2Ind = {}



    # ------------------

    # Write your implementation here.

    for i in range(num_words):

        word2Ind[words[i]] = i

    M = np.zeros((num_words, num_words))

    for line in corpus:

        for i in range(len(line)):

            target = line[i]

            target_index = word2Ind[target]

            

            left = max(i - window_size, 0)

            right = min(i + window_size, len(line) - 1)

            for j in range(left, i):

                window_word = line[j]

                M[target_index][word2Ind[window_word]] += 1

                M[word2Ind[window_word]][target_index] += 1







    # ------------------



    return M, word2Ind







# ---------------------

# Run this sanity check

# Note that this is not an exhaustive check for correctness.

# ---------------------



# Define toy corpus and get student's co-occurrence matrix

test_corpus = ["START All that glitters isn't gold END".split(" "), "START All's well that ends well END".split(" ")]

M_test, word2Ind_test = compute_co_occurrence_matrix(test_corpus, window_size=1)



# Correct M and word2Ind

M_test_ans = np.array(

    [[0., 0., 0., 1., 0., 0., 0., 0., 1., 0.,],

     [0., 0., 0., 1., 0., 0., 0., 0., 0., 1.,],

     [0., 0., 0., 0., 0., 0., 1., 0., 0., 1.,],

     [1., 1., 0., 0., 0., 0., 0., 0., 0., 0.,],

     [0., 0., 0., 0., 0., 0., 0., 0., 1., 1.,],

     [0., 0., 0., 0., 0., 0., 0., 1., 1., 0.,],

     [0., 0., 1., 0., 0., 0., 0., 1., 0., 0.,],

     [0., 0., 0., 0., 0., 1., 1., 0., 0., 0.,],

     [1., 0., 0., 0., 1., 1., 0., 0., 0., 1.,],

     [0., 1., 1., 0., 1., 0., 0., 0., 1., 0.,]]

)

word2Ind_ans = {'All': 0, "All's": 1, 'END': 2, 'START': 3, 'ends': 4, 'glitters': 5, 'gold': 6, "isn't": 7, 'that': 8, 'well': 9}



# Test correct word2Ind

assert (word2Ind_ans == word2Ind_test), "Your word2Ind is incorrect:\nCorrect: {}\nYours: {}".format(word2Ind_ans, word2Ind_test)



# Test correct M shape

assert (M_test.shape == M_test_ans.shape), "M matrix has incorrect shape.\nCorrect: {}\nYours: {}".format(M_test.shape, M_test_ans.shape)



# Test correct M values

for w1 in word2Ind_ans.keys():

    idx1 = word2Ind_ans[w1]

    for w2 in word2Ind_ans.keys():

        idx2 = word2Ind_ans[w2]

        student = M_test[idx1, idx2]

        correct = M_test_ans[idx1, idx2]

        if student != correct:

            print("Correct M:")

            print(M_test_ans)

            print("Your M: ")

            print(M_test)

            raise AssertionError("Incorrect count at index ({}, {})=({}, {}) in matrix M. Yours has {} but should have {}.".format(idx1, idx2, w1, w2, student, correct))



# Print Success

print ("-" * 80)

print("Passed All Tests for Q1.2!")

print ("-" * 80)



#=================================================================================================================================

#Question 1.3: Implement reduce_to_k_dim [code] (1 point)



from sklearn.decomposition import TruncatedSVD

from sklearn.random_projection import sparse_random_matrix





def reduce_to_k_dim(M, k=2):

    """ Reduce a co-occurence count matrix of dimensionality (num_corpus_words, num_corpus_words)

        to a matrix of dimensionality (num_corpus_words, k) using the following SVD function from Scikit-Learn:

            - http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html



        Params:

            M (numpy matrix of shape (number of corpus words, number of corpus words)): co-occurence matrix of word counts

            k (int): embedding size of each word after dimension reduction

        Return:

            M_reduced (numpy matrix of shape (number of corpus words, k)): matrix of k-dimensioal word embeddings.

                    In terms of the SVD from math class, this actually returns U * S

    """

    n_iters = 10  # Use this parameter in your call to `TruncatedSVD`

    M_reduced = None

    print("Running Truncated SVD over %i words..." % (M.shape[0]))



    # ------------------

    # Write your implementation here.

    from sklearn.decomposition import TruncatedSVD

    svd = TruncatedSVD(n_components=k, n_iter=n_iters)

    svd.fit(M)

    M_reduced = svd.transform(M)







    # ------------------



    print("Done.")

    return M_reduced



#==========================================================================================================================



# ---------------------

# Run this sanity check

# Note that this not an exhaustive check for correctness

# In fact we only check that your M_reduced has the right dimensions.

# ---------------------



# Define toy corpus and run student code

test_corpus = ["START All that glitters isn't gold END".split(" "), "START All's well that ends well END".split(" ")]

M_test, word2Ind_test = compute_co_occurrence_matrix(test_corpus, window_size=1)

M_test_reduced = reduce_to_k_dim(M_test, k=2)



# Test proper dimensions

assert (M_test_reduced.shape[0] == 10), "M_reduced has {} rows; should have {}".format(M_test_reduced.shape[0], 10)

assert (M_test_reduced.shape[1] == 2), "M_reduced has {} columns; should have {}".format(M_test_reduced.shape[1], 2)



# Print Success

print ("-" * 80)

print("Passed All Tests for Q1.3!")

print ("-" * 80)





#Q1.3 Ends here

#===========================================================================================================================





#======================================================================================================================

#Question 1.4: Implement plot_embeddings [code] (1 point)

def plot_embeddings(M_reduced, word2Ind, words):

    """ Plot in a scatterplot the embeddings of the words specified in the list "words".

        NOTE: do not plot all the words listed in M_reduced / word2Ind.

        Include a label next to each point.



        Params:

            M_reduced (numpy matrix of shape (number of unique words in the corpus , k)): matrix of k-dimensioal word embeddings

            word2Ind (dict): dictionary that maps word to indices for matrix M

            words (list of strings): words whose embeddings we want to visualize

    """

    

    # ------------------

    # Write your implementation here.

    words_index = [word2Ind[word] for word in words]

    print("Word Index = ", words_index)

    x_coords = [M_reduced[word_index][0] for word_index in words_index]

    y_coords = [M_reduced[word_index][1] for word_index in words_index]

    

    for i, word in enumerate(words):

        x = x_coords[i]

        y = y_coords[i]

        plt.scatter(x, y, marker = 'x', color = 'red')

        plt.text(x + 0.0003, y + 0.0003, word, fontsize = 9)

        

    



    # ------------------



    # ---------------------

    # Run this sanity check

    # Note that this not an exhaustive check for correctness.

    # The plot produced should look like the "test solution plot" depicted below.

    # ---------------------



#The Answer ends here

#============================================================================================================================







#Q1.5

# -----------------------------

# Run This Cell to Produce Your Plot

# ------------------------------

reuters_corpus = read_corpus()

M_co_occurrence, word2Ind_co_occurrence = compute_co_occurrence_matrix(reuters_corpus)

M_reduced_co_occurrence = reduce_to_k_dim(M_co_occurrence, k=2)



# Rescale (normalize) the rows to make them each of unit-length

M_lengths = np.linalg.norm(M_reduced_co_occurrence, axis=1)

M_normalized = M_reduced_co_occurrence / M_lengths[:, np.newaxis] # broadcasting



words = ['barrels', 'bpd', 'ecuador', 'energy', 'industry', 'kuwait', 'oil', 'output', 'petroleum', 'venezuela']

plot_embeddings(M_normalized, word2Ind_co_occurrence, words)







#============================================================================================================================



#Part 2 begins

def load_word2vec():

    """ Load Word2Vec Vectors

        Return:

            wv_from_bin: All 3 million embeddings, each lengh 300

    """

    from gensim.models import Word2Vec

    import gensim

  

    print("gensim imported")

    # Load Google's pre-trained Word2Vec model.





    wv_from_bin = gensim.models.KeyedVectors.load_word2vec_format("../input/GoogleNews-vectors-negative300.bin", binary=True)

    print("model trained")



    vocab = list(wv_from_bin.vocab.keys())

    print("Loaded vocab size %i" % len(vocab))

    return wv_from_bin



# -----------------------------------

# Run Cell to Load Word Vectors

# Note: This may take several minutes

# -----------------------------------

wv_from_bin = load_word2vec()

def get_matrix_of_vectors(wv_from_bin, required_words=['barrels', 'bpd', 'ecuador', 'energy', 'industry', 'kuwait', 'oil', 'output', 'petroleum', 'venezuela']):

    """ Put the word2vec vectors into a matrix M.

        Param:

            wv_from_bin: KeyedVectors object; the 3 million word2vec vectors loaded from file

        Return:

            M: numpy matrix shape (num words, 300) containing the vectors

            word2Ind: dictionary mapping each word to its row number in M

    """

    import random

    words = list(wv_from_bin.vocab.keys())

    print("Shuffling words ...")

    random.shuffle(words)

    words = words[:10000]

    print("Putting %i words into word2Ind and matrix M..." % len(words))

    word2Ind = {}

    M = []

    curInd = 0

    for w in words:

        try:

            M.append(wv_from_bin.word_vec(w))

            word2Ind[w] = curInd

            curInd += 1

        except KeyError:

            continue

    for w in required_words:

        try:

            M.append(wv_from_bin.word_vec(w))

            word2Ind[w] = curInd

            curInd += 1

        except KeyError:

            continue

    M = np.stack(M)

    print("Done.")

    return M, word2Ind
# -----------------------------------------------------------------

# Run Cell to Reduce 300-Dimensinal Word Embeddings to k Dimensions

# Note: This may take several minutes

# -----------------------------------------------------------------

M, word2Ind = get_matrix_of_vectors(wv_from_bin)

M_reduced = reduce_to_k_dim(M, k=2)
words = ['barrels', 'bpd', 'ecuador', 'energy', 'industry', 'kuwait', 'oil', 'output', 'petroleum', 'venezuela']

plot_embeddings(M_reduced, word2Ind, words)
# ------------------

# Write your polysemous word exploration code here.



wv_from_bin.most_similar("lie")



# ------------------
# ------------------

# Write your synonym & antonym exploration code here.



w1 = "angel"

w2 = "saint"

w3 = "demon"

w1_w2_dist = wv_from_bin.distance(w1, w2)

w1_w3_dist = wv_from_bin.distance(w1, w3)



print("Synonyms {}, {} have cosine distance: {}".format(w1, w2, w1_w2_dist))

print("Antonyms {}, {} have cosine distance: {}".format(w1, w3, w1_w3_dist))



# ------------------
# Run this cell to answer the analogy -- man : king :: woman : x

pprint.pprint(wv_from_bin.most_similar(positive=['woman', 'king'], negative=['man']))
# ------------------

# Write your analogy exploration code here.



pprint.pprint(wv_from_bin.most_similar(positive=['father','woman'], negative=['man']))



# ------------------
# ------------------

# Write your incorrect analogy exploration code here.



pprint.pprint(wv_from_bin.most_similar(positive=['apple','orange'], negative=['red']))



# ------------------
# Run this cell

# Here `positive` indicates the list of words to be similar to and `negative` indicates the list of words to be

# most dissimilar from.

pprint.pprint(wv_from_bin.most_similar(positive=['woman', 'boss'], negative=['man']))

print()

pprint.pprint(wv_from_bin.most_similar(positive=['man', 'boss'], negative=['woman']))
# ------------------

# Write your bias exploration code here.



pprint.pprint(wv_from_bin.most_similar(positive=['doctor','female'], negative=['male']))

print()

pprint.pprint(wv_from_bin.most_similar(positive=['nurse','male'], negative=['female']))



# ------------------