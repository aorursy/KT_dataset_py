# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.ensemble import RandomForestClassifier

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# Import the pandas package, then use the "read_csv" function to read

# the labeled training data

      

train = pd.read_csv("/kaggle/input/word2vec-nlp-tutorial/labeledTrainData.tsv.zip", header=0, \

                    delimiter="\t", quoting=3)
test = pd.read_csv("/kaggle/input/word2vec-nlp-tutorial/testData.tsv.zip", header=0, \

                    delimiter="\t", quoting=3 )
# Here, "header=0" indicates that the first line of the file contains column names, 

# "delimiter=\t" indicates that the fields are separated by tabs, and quoting=3 tells Python to ignore doubled quotes, 

# otherwise you may encounter errors trying to read the file.
train.shape
train.columns.values
print(train["review"][0])
import re

import nltk



from bs4 import BeautifulSoup

from nltk.corpus import stopwords





class KaggleWord2VecUtility(object):

    """KaggleWord2VecUtility is a utility class for processing raw HTML text into segments for further learning"""



    @staticmethod

    def review_to_wordlist( review, remove_stopwords=False ):

        # Function to convert a document to a sequence of words,

        # optionally removing stop words.  Returns a list of words.

        #

        # 1. Remove HTML

        review_text = BeautifulSoup(review).get_text()

        #

        # 2. Remove non-letters

        review_text = re.sub("[^a-zA-Z]"," ", review_text)

        #

        # 3. Convert words to lower case and split them

        words = review_text.lower().split()

        #

        # 4. Optionally remove stop words (false by default)

        if remove_stopwords:

            stops = set(stopwords.words("english"))

            words = [w for w in words if not w in stops]

        #

        # 5. Return a list of words

        return(words)



    # Define a function to split a review into parsed sentences

    @staticmethod

    def review_to_sentences( review, tokenizer, remove_stopwords=False ):

        # Function to split a review into parsed sentences. Returns a

        # list of sentences, where each sentence is a list of words

        #

        # 1. Use the NLTK tokenizer to split the paragraph into sentences

        raw_sentences = tokenizer.tokenize(review.strip())

        #

        # 2. Loop over each sentence

        sentences = []

        for raw_sentence in raw_sentences:

            # If a sentence is empty, skip it

            if len(raw_sentence) > 0:

                # Otherwise, call review_to_wordlist to get a list of words

                sentences.append( KaggleWord2VecUtility.review_to_wordlist( raw_sentence, \

                  remove_stopwords ))

        #

        # Return the list of sentences (each sentence is a list of words,

        # so this returns a list of lists

        return sentences
# Initialize an empty list to hold the clean reviews

clean_train_reviews = []



# Loop over each review; create an index i that goes from 0 to the length

# of the movie review list



print ("Cleaning and parsing the training set movie reviews...\n")

for i in range( 0, len(train["review"])):

    clean_train_reviews.append(" ".join(KaggleWord2VecUtility.review_to_wordlist(train["review"][i], True)))
clean_train_reviews[0]
len(clean_train_reviews)
# Initialize the "CountVectorizer" object, which is scikit-learn's

# bag of words tool.

countVectorizer = CountVectorizer(analyzer = "word",   \

                         tokenizer = None,    \

                         preprocessor = None, \

                         stop_words = None,   \

                         max_features = 5000)



# fit_transform() does two functions: First, it fits the model

# and learns the vocabulary; second, it transforms our training data

# into feature vectors. The input to fit_transform should be a list of

# strings.

train_data_features = countVectorizer.fit_transform(clean_train_reviews)

type(train_data_features)
# Numpy arrays are easy to work with, so convert the result to an

# array

np.asarray(train_data_features)
print ("Training the random forest (this may take a while)...")





# Initialize a Random Forest classifier with 100 trees

forest = RandomForestClassifier(n_estimators = 100)



# Fit the forest to the training set, using the bag of words as

# features and the sentiment labels as the response variable

#

# This may take a few minutes to run

forest = forest.fit( train_data_features, train["sentiment"] )


# Testing - we pick only the first 2 test reviews alone to save time





# Create an empty list and append the clean reviews one by one

clean_test_reviews = []



print("Cleaning and parsing the test set movie reviews...\n")

for i in range(0,2):

    clean_test_reviews.append(" ".join(KaggleWord2VecUtility.review_to_wordlist(test["review"][i], True)))



# Get a bag of words for the test set, and convert to a numpy array

test_data_features = countVectorizer.transform(clean_test_reviews)

np.asarray(test_data_features)



# Use the random forest to make sentiment label predictions

print ("Predicting test labels...\n")

result = forest.predict(test_data_features)



# Write the test results 

print("*****manual verification*******")

print(test["review"][0])

print(result[0])

print(test["review"][1])

print(result[1])



# Copy the results to a pandas dataframe with an "id" column and

# a "sentiment" column

# output = pd.DataFrame( data={"id":test["id"], "sentiment":result} )



# Use pandas to write the comma-separated output file

# output.to_csv(os.path.join(os.path.dirname(__file__), 'data', 'Bag_of_Words_model.csv'), index=False, quoting=3)#

# print ("Wrote results to Bag_of_Words_model.csv")
type(output)
output.head(1)
from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.pipeline import Pipeline

pipe = Pipeline([('count', countVectorizer),('tfid', TfidfTransformer())]).fit(clean_train_reviews)

train_data_features = pipe.transform(clean_train_reviews)
type(train_data_features)
train_data_features.shape
np.asarray(train_data_features)
print ("Training the random forest (this may take a while)...")





# Initialize a Random Forest classifier with 100 trees

forest = RandomForestClassifier(n_estimators = 100)



# Fit the forest to the training set, using the bag of words as

# features and the sentiment labels as the response variable

#

# This may take a few minutes to run

forest = forest.fit( train_data_features, train["sentiment"] )
# Testing - we pick only the first 2 test reviews alone to save time





# Create an empty list and append the clean reviews one by one

clean_test_reviews = []



print("Cleaning and parsing the test set movie reviews...\n")

for i in range(0,2):

    clean_test_reviews.append(" ".join(KaggleWord2VecUtility.review_to_wordlist(test["review"][i], True)))



# Get a bag of words for the test set, and convert to a numpy array

test_data_features = countVectorizer.transform(clean_test_reviews)

np.asarray(test_data_features)



# Use the random forest to make sentiment label predictions

print ("Predicting test labels...\n")

result = forest.predict(test_data_features)



# Write the test results 

print("*****manual verification*******")

print(test["review"][0])

print(result[0])

print(test["review"][1])

print(result[1])

import os

from nltk.corpus import stopwords

import nltk.data

import logging

from gensim.models import Word2Vec

from sklearn.ensemble import RandomForestClassifier
unlabeled_train = pd.read_csv("/kaggle/input/word2vec-nlp-tutorial/unlabeledTrainData.tsv.zip", header=0, \

                    delimiter="\t", quoting=3)
# Verify the number of reviews that were read (100,000 in total)

print ("Read %d labeled train reviews, %d labeled test reviews, " \

 "and %d unlabeled reviews\n" % (train["review"].size,

 test["review"].size, unlabeled_train["review"].size ))
# Load the punkt tokenizer

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
unlabeled_train["review"].size
%%capture --no-stderr --no-stdout



# ****** Split the labeled and unlabeled training sets into clean sentences

#

sentences = []  # Initialize an empty list of sentences



#I prefer not adding the sentenses from labeled training set because then it takes a vary long time to run

#print ("Parsing sentences from training set")

#for review in train["review"]:

#    sentences += KaggleWord2VecUtility.review_to_sentences(review, tokenizer)



print ("Parsing sentences from unlabeled set")

for review in unlabeled_train["review"]:

    sentences += KaggleWord2VecUtility.review_to_sentences(review, tokenizer)
# Check how many sentences we have in total - should be around 50,000+

print (len(sentences))
sentences[0]
# ****** Set parameters and train the word2vec model

#

# Import the built-in logging module and configure it so that Word2Vec

# creates nice output messages

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',\

    level=logging.INFO)



# Set values for various parameters

num_features = 300    # Word vector dimensionality

min_word_count = 40   # Minimum word count

num_workers = 4       # Number of threads to run in parallel

context = 10          # Context window size

downsampling = 1e-3   # Downsample setting for frequent words



# Initialize and train the model (this will take some time)

print ("Training Word2Vec model...")

model = Word2Vec(sentences, workers=num_workers, \

            size=num_features, min_count = min_word_count, \

            window = context, sample = downsampling, seed=1)
# If you don't plan to train the model any further, calling

# init_sims will make the model much more memory-efficient.

model.init_sims(replace=True)



# It can be helpful to create a meaningful model name and

# save the model for later use. You can load it later using Word2Vec.load()

model_name = "300features_40minwords_10context"

model.save(model_name)
model.wv.doesnt_match("man woman child kitchen".split())
model.wv.doesnt_match("france england germany berlin".split())
model.wv.most_similar("man")
# Load the model that we created in Part 2

from gensim.models import Word2Vec

model = Word2Vec.load("300features_40minwords_10context")
type(model.wv.vectors)
model.wv.vectors.shape
model.wv.__getitem__("flower") #1x300 numpy array
def makeFeatureVec(words, model, num_features):

    # Function to average all of the word vectors in a given

    # paragraph

    #

    # Pre-initialize an empty numpy array (for speed)

    featureVec = np.zeros((num_features,),dtype="float32")

    #

    nwords = 0.

    # 

    # Index2word is a list that contains the names of the words in 

    # the model's vocabulary. Convert it to a set, for speed 

    index2word_set = set(model.wv.index2word)



    #

    # Loop over each word in the review and, if it is in the model's

    # vocaublary, add its feature vector to the total

    for word in words:

        if word in index2word_set: 

            nwords = nwords + 1.

            featureVec = np.add(featureVec,model.wv.__getitem__([word]))



    print(featureVec.shape)

    print(nwords)

    # Divide the result by the number of words to get the average

    featureVec = np.divide(featureVec,nwords)

    return featureVec





def getAvgFeatureVecs(reviews, model, num_features):

    # Given a set of reviews (each one a list of words), calculate 

    # the average feature vector for each one and return a 2D numpy array 

    # 

    # Initialize a counter

    counter = 0

    # 

    # Preallocate a 2D numpy array, for speed

    reviewFeatureVecs = np.zeros((len(reviews),num_features),dtype="float32")

    print(reviewFeatureVecs.shape)

    

    # 

    # Loop through the reviews

    for review in reviews:

        #

        # Print a status message every 1000th review

        print ("Review %d of %d" % (counter, len(reviews)))

        # 

        # Call the function (defined above) that makes average feature vectors

        print("Fetching avg vector for review {0}".format(review))

        reviewFeatureVecs[counter] = makeFeatureVec(review, model, num_features)



        print("**********reviewFeatureVec for review %d ***************" %(counter))

        print(reviewFeatureVecs[counter])

        print("*************************")

        #

        # Increment the counter

        counter = counter + 1

    return reviewFeatureVecs
train['review'][10]
train['review'][23]
# this is a very toy example, to test the above functions

review_docs=[train['review'][10], train['review'][23]]



clean_docs = []

for review in review_docs:

    clean_docs.append(KaggleWord2VecUtility.review_to_wordlist(review, remove_stopwords=True ))



trainDataVecs = getAvgFeatureVecs( clean_docs, model, num_features )
def makeFeatureVec(words, model, num_features):

    # Function to average all of the word vectors in a given

    # paragraph

    #

    # Pre-initialize an empty numpy array (for speed)

    featureVec = np.zeros((num_features,),dtype="float32")

    #

    nwords = 0.

    # 

    # Index2word is a list that contains the names of the words in 

    # the model's vocabulary. Convert it to a set, for speed 

    index2word_set = set(model.wv.index2word)



    #

    # Loop over each word in the review and, if it is in the model's

    # vocaublary, add its feature vector to the total

    for word in words:

        if word in index2word_set: 

            nwords = nwords + 1.

            featureVec = np.add(featureVec,model.wv.__getitem__([word]))



    # Divide the result by the number of words to get the average

    featureVec = np.divide(featureVec,nwords)

    return featureVec





def getAvgFeatureVecs(reviews, model, num_features):

    # Given a set of reviews (each one a list of words), calculate 

    # the average feature vector for each one and return a 2D numpy array 

    # 

    # Initialize a counter

    counter = 0

    # 

    # Preallocate a 2D numpy array, for speed

    reviewFeatureVecs = np.zeros((len(reviews),num_features),dtype="float32")

    

    # 

    # Loop through the reviews

    for review in reviews:

        #

        # Print a status message every 1000th review

        if counter%1000. == 0.:

            print ("Review %d of %d" % (counter, len(reviews)))

        # 

        # Call the function (defined above) that makes average feature vectors

        reviewFeatureVecs[counter] = makeFeatureVec(review, model, num_features)

        #

        # Increment the counter

        counter = counter + 1

    return reviewFeatureVecs
# ****************************************************************

# Calculate average feature vectors for training and testing sets,

# using the functions we defined above. Notice that we now use stop word

# removal.



clean_train_reviews = []

for review in train["review"]:

    clean_train_reviews.append( KaggleWord2VecUtility.review_to_wordlist( review, remove_stopwords=True ))



trainDataVecs = getAvgFeatureVecs( clean_train_reviews, model, num_features )

# Fit a random forest to the training data, using 100 trees

from sklearn.ensemble import RandomForestClassifier

forest = RandomForestClassifier( n_estimators = 100 )



print ("Fitting a random forest to labeled training data...")

forest = forest.fit( trainDataVecs, train["sentiment"] )
# Testing - we pick only the first 2 test reviews alone to save time



# First we need to construct avg feature vecs

print ("Creating average feature vecs for test reviews")

clean_test_reviews = []

for review in (test["review"][0], test["review"][1]):

    clean_test_reviews.append( KaggleWord2VecUtility.review_to_wordlist( review, remove_stopwords=True ))



testDataVecs = getAvgFeatureVecs( clean_test_reviews, model, num_features )



# Test & extract results 

result = forest.predict( testDataVecs )



# Write the test results 

print("*****manual verification*******")

print(test["review"][0])

print(result[0])

print(test["review"][1])

print(result[1])
from sklearn.cluster import KMeans

import time



start = time.time() # Start time



# Set "k" (num_clusters) to be 1/5th of the vocabulary size, or an

# average of 5 words per cluster

# But I set it to 1/100th to make it run fast

word_vectors = model.wv.vectors

num_clusters = int(word_vectors.shape[0] / 100)



# Initalize a k-means object and use it to extract centroids

kmeans_clustering = KMeans( n_clusters = num_clusters )

idx = kmeans_clustering.fit_predict( word_vectors )



# Get the end time and print how long the process took

end = time.time()

elapsed = end - start

print ("Time taken for K Means clustering: %10.2f seconds." %(elapsed))
# Create a Word / Index dictionary, mapping each vocabulary word to

# a cluster number                                                                                            

word_centroid_map = dict(zip( model.wv.index2word, idx ))
# For the first 5 clusters

for cluster in range(0, 5):

    #

    # Print the cluster number  

    print ("\nCluster %d" % cluster)

    #

    # Find all of the words for that cluster number, and print them out

    words = []

    for i in range(0,len(word_centroid_map.values())):

        if( list(word_centroid_map.values())[i] == cluster ):

            words.append(list(word_centroid_map.keys())[i])

    print (words)
def create_bag_of_centroids( wordlist, word_centroid_map ):

    #

    # The number of clusters is equal to the highest cluster index

    # in the word / centroid map

    num_centroids = max( word_centroid_map.values() ) + 1

    #

    # Pre-allocate the bag of centroids vector (for speed)

    bag_of_centroids = np.zeros( num_centroids, dtype="float32" )

    #

    # Loop over the words in the review. If the word is in the vocabulary,

    # find which cluster it belongs to, and increment that cluster count 

    # by one

    for word in wordlist:

        if word in word_centroid_map:

            index = word_centroid_map[word]

            bag_of_centroids[index] += 1

    #

    # Return the "bag of centroids"

    return bag_of_centroids
# Pre-allocate an array for the training set bags of centroids (for speed)

train_centroids = np.zeros( (train["review"].size, num_clusters), \

    dtype="float32" )



# Transform the training set reviews into bags of centroids

counter = 0

for review in clean_train_reviews:

    train_centroids[counter] = create_bag_of_centroids( review, \

        word_centroid_map )

    counter += 1

# Fit a random forest and extract predictions 

forest = RandomForestClassifier(n_estimators = 100)



# Fitting the forest may take a few minutes

print ("Fitting a random forest to labeled training data...")

forest = forest.fit(train_centroids,train["sentiment"])
# Testing - we pick only the first 2 test reviews alone to save time



#hard coding 2 - should use test["review"].size

test_centroids = np.zeros(( 2, num_clusters), \

    dtype="float32" )



# First we need to construct avg feature vecs

print ("Creating average feature vecs for test reviews")

clean_test_reviews = []

for review in (test["review"][0], test["review"][1]):

    clean_test_reviews.append( KaggleWord2VecUtility.review_to_wordlist( review, remove_stopwords=True ))



counter = 0

for review in clean_test_reviews:

    test_centroids[counter] = create_bag_of_centroids( review, \

        word_centroid_map )

    counter += 1



# Test & extract results 

result = forest.predict( test_centroids )



# Write the test results 

print("*****manual verification*******")

print(test["review"][0])

print(result[0])

print(test["review"][1])

print(result[1])