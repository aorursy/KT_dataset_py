# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#Import libraries

from bs4 import BeautifulSoup as bs

import re

import nltk.data

from nltk.corpus import stopwords

import logging

from gensim.models import word2vec, Word2Vec

from sklearn.cluster import KMeans

import time
# Read data from files 

train = pd.read_csv("../input/word2vec-nlp-tutorial/labeledTrainData.tsv.zip", header=0, delimiter="\t", quoting=3 )

test = pd.read_csv("../input/word2vec-nlp-tutorial/testData.tsv.zip", header=0, delimiter="\t", quoting=3 )

unlabeled_train = pd.read_csv("../input/word2vec-nlp-tutorial/unlabeledTrainData.tsv.zip", header=0, delimiter="\t", quoting=3 )
# Verify the number of reviews that were read (100,000 in total)

print(f"Read {train['review'].size} labeled train reviews, {test['review'].size} labeled test reviews and {unlabeled_train['review'].size} unlabeled reviews")
def review_to_wordlist(review, remove_stopwords=False):

    '''Function to convert a document to a sequence of words,

    optionally removing stop words.  Returns a list of words.

    '''

    # 1. Remove HTML

    review_text = bs(review).get_text()

      

    # 2. Remove non-letters

    review_text = re.sub("[^a-zA-Z]"," ", review_text)

    

    # 3. Convert words to lower case and split them

    words = review_text.lower().split()

    

    # 4. Optionally remove stop words (false by default)

    if remove_stopwords:

        stops = set(stopwords.words("english"))

        words = [w for w in words if not w in stops]

    

    # 5. Return a list of words

    return(words)
# Download the punkt tokenizer for sentence splitting



#nltk.download()
# Load the punkt tokenizer

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
# Define a function to split a review into parsed sentences

def review_to_sentences(review, tokenizer, remove_stopwords=False):

    '''Function to split a review into parsed sentences. Returns a 

    list of sentences, where each sentence is a list of words

    '''

    # 1. Use the NLTK tokenizer to split the paragraph into sentences

    raw_sentences = tokenizer.tokenize(review.strip())

    

    # 2. Loop over each sentence

    sentences = []

    for raw_sentence in raw_sentences:

        # If a sentence is empty, skip it

        if len(raw_sentence) > 0:

            # Otherwise, call review_to_wordlist to get a list of words

            sentences.append(review_to_wordlist(raw_sentence, remove_stopwords))

    

    # Return the list of sentences (each sentence is a list of words,

    # so this returns a list of lists

    return sentences
sentences = []  # Initialize an empty list of sentences



print("Parsing sentences from training set")

for review in train["review"]:

    sentences += review_to_sentences(review, tokenizer)

print("Finished parsing sentences from training set")



print("Parsing sentences from unlabeled set")

for review in unlabeled_train["review"]:

    sentences += review_to_sentences(review, tokenizer)

print("Finished parsing sentences from unlabeled set")
# Import the built-in logging module and configure it so that Word2Vec 

# creates nice output messages

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
# Set values for various parameters

num_features = 300    # Word vector dimensionality                      

min_word_count = 40   # Minimum word count                        

num_workers = 4       # Number of threads to run in parallel

context = 10          # Context window size                                                                                    

downsampling = 1e-3   # Downsample setting for frequent words
# Initialize and train the model (this will take some time)

print("Training model...")

model = word2vec.Word2Vec(sentences,

                          workers=num_workers,

                          size=num_features,

                          min_count = min_word_count,

                          window = context,

                          sample = downsampling)

print("Finished training model.")
# If you don't plan to train the model any further, calling 

# init_sims will make the model much more memory-efficient.

model.init_sims(replace=True)
# It can be helpful to create a meaningful model name and 

# save the model for later use. You can load it later using Word2Vec.load()

model_name = "300features_40minwords_10context"

model.save(model_name)
# Load the model that we created in Part 2

model = Word2Vec.load("../input/part2-model/300features_40minwords_10context")
type(model)
model.wv.vectors
model.wv.vectors.shape
model.wv["flower"]
type(model.wv["flower"])
model.wv["flower"].shape
model.wv.most_similar("man")
def makeFeatureVec(words, model, num_features):

    '''Function to average all of the word vectors in a given

    paragraph'''

    

    # Pre-initialize an empty numpy array (for speed)

    featureVec = np.zeros((num_features,),dtype="float32")

    nwords = 0.

     

    # Index2word is a list that contains the names of the words in 

    # the model's vocabulary. Convert it to a set, for speed 

    index2word_set = set(model.wv.index2word)

    

    # Loop over each word in the review and, if it is in the model's

    # vocaublary, add its feature vector to the total

    for word in words:

        if word in index2word_set: 

            nwords = nwords + 1.

            featureVec = np.add(featureVec,model[word])

     

    # Divide the result by the number of words to get the average

    featureVec = np.divide(featureVec,nwords)

    return featureVec
def getAvgFeatureVecs(reviews, model, num_features):

    '''Given a set of reviews (each one a list of words), calculate 

    the average feature vector for each one and return a 2D numpy array '''

     

    # Initialize a counter

    counter = 0

    

    # Preallocate a 2D numpy array, for speed

    reviewFeatureVecs = np.zeros((len(reviews),num_features),dtype="float32")

    

    # Loop through the reviews

    for review in reviews:

       # Print a status message every 1000th review

       if counter%1000 == 0:

            print(f"Review {counter} of {len(reviews)}")

            

       # Call the function (defined above) that makes average feature vectors

       reviewFeatureVecs[counter] = makeFeatureVec(review, model, num_features)

        

       # Increment the counter

       counter = counter + 1

        

    return reviewFeatureVecs
# Calculate average feature vectors for training and testing sets,

# using the functions we defined above. Notice that we now use stop word

# removal.



clean_train_reviews = []

for review in train["review"]:

    clean_train_reviews.append(review_to_wordlist(review, remove_stopwords=True))



trainDataVecs = getAvgFeatureVecs(clean_train_reviews, model, num_features)



print("Creating average feature vecs for test reviews")

clean_test_reviews = []

for review in test["review"]:

    clean_test_reviews.append( review_to_wordlist(review,remove_stopwords=True))

print("Finished creating average feature vecs for test reviews")



testDataVecs = getAvgFeatureVecs(clean_test_reviews, model, num_features)
# Fit a random forest to the training data, using 100 trees

from sklearn.ensemble import RandomForestClassifier

forest = RandomForestClassifier(n_estimators = 100)



print("Fitting a random forest to labeled training data...")

forest = forest.fit(trainDataVecs, train["sentiment"])

print("Finished fitting a random forest to labeled training data...")



# Test & extract results 

result = forest.predict(testDataVecs)



# Write the test results 

output = pd.DataFrame(data={"id":test["id"], "sentiment":result})

output.to_csv("Word2Vec_AverageVectors.csv", index=False, quoting=3)
start = time.time() # Start time



# Set "k" (num_clusters) to be 1/5th of the vocabulary size, or an

# average of 5 words per cluster

word_vectors = model.wv.vectors

num_clusters = int(word_vectors.shape[0] / 5)



# Initalize a k-means object and use it to extract centroids

kmeans_clustering = KMeans(n_clusters = num_clusters)

idx = kmeans_clustering.fit_predict(word_vectors)



# Get the end time and print how long the process took

end = time.time()

elapsed = end - start

print("Time taken for K Means clustering: ", elapsed, "seconds.")
# Create a Word / Index dictionary, mapping each vocabulary word to

# a cluster number                                                                                            

word_centroid_map = dict(zip(model.wv.index2word, idx))

word_centroid_map
# For the first 10 clusters

for cluster in range(0,10):

    

    # Print the cluster number  

    print(f"\nCluster {cluster}")

    

    # Find all of the words for that cluster number, and print them out

    words = []

    for i in range(0,len(word_centroid_map.values())):

        if(list(word_centroid_map.values())[i] == cluster):

            words.append(list(word_centroid_map.keys())[i])

    print(words)
def create_bag_of_centroids( wordlist, word_centroid_map ):

    

    # The number of clusters is equal to the highest cluster index

    # in the word / centroid map

    num_centroids = max(word_centroid_map.values())+1

    

    # Pre-allocate the bag of centroids vector (for speed)

    bag_of_centroids = np.zeros(num_centroids, dtype="float32")

    

    # Loop over the words in the review. If the word is in the vocabulary,

    # find which cluster it belongs to, and increment that cluster count 

    # by one

    for word in wordlist:

        if word in word_centroid_map:

            index = word_centroid_map[word]

            bag_of_centroids[index] += 1

    

    # Return the "bag of centroids"

    return bag_of_centroids
# Pre-allocate an array for the training set bags of centroids (for speed)

train_centroids = np.zeros((train["review"].size, num_clusters),dtype="float32")



# Transform the training set reviews into bags of centroids

counter = 0

for review in clean_train_reviews:

    train_centroids[counter] = create_bag_of_centroids(review, word_centroid_map)

    counter += 1



# Repeat for test reviews 

test_centroids = np.zeros((test["review"].size, num_clusters), dtype="float32")



counter = 0

for review in clean_test_reviews:

    test_centroids[counter] = create_bag_of_centroids(review, word_centroid_map)

    counter += 1
# Fit a random forest and extract predictions 

forest = RandomForestClassifier(n_estimators = 100)



# Fitting the forest may take a few minutes

print("Fitting a random forest to labeled training data...")

forest = forest.fit(train_centroids,train["sentiment"])

result = forest.predict(test_centroids)

print("Finished fitting a random forest to labeled training data.")



# Write the test results 

output = pd.DataFrame(data={"id":test["id"], "sentiment":result})

output.to_csv( "BagOfCentroids.csv", index=False, quoting=3 )