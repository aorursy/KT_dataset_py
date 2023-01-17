import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline
training_data = pd.read_csv('../input/word2vec-nlp-tutorial/labeledTrainData.tsv',header=0,delimiter='\t',quoting=3)
training_data.head()
training_data['review'][0]
#it is not considered a reliable practice to remove markup using regular expressions,

#so even for an application as simple as this, it's usually best to use a package like BeautifulSoup.

#removing HTML Markups and Tags like "<br>"

from bs4 import BeautifulSoup  
example1 = BeautifulSoup(training_data["review"][0]) 
example1.get_text()
#creating a function to clean the reviews

def review_to_words( raw_review ):

    # Function to convert a raw review to a string of words

    # The input is a single string (a raw movie review), and 

    # the output is a single string (a preprocessed movie review)

    #

    # 1. Remove HTML

    review_text = BeautifulSoup(raw_review).get_text() 

    #

    # 2. Remove non-letters        

    letters_only = re.sub("[^a-zA-Z]", " ", review_text) 

    #

    # 3. Convert to lower case, split into individual words

    words = letters_only.lower().split()                             

    #

    # 4. In Python, searching a set is much faster than searching

    #   a list, so convert the stop words to a set

    stops = set(stopwords.words("english"))                  

    # 

    # 5. Remove stop words

    meaningful_words = [w for w in words if not w in stops]   

    

    #6.Lemmatization

    for word in meaningful_words:

        word = wordnet_lemmatizer.lemmatize(word,'v')

    

    # 6. Join the words back into one string separated by space, 

    # and return the result.

    return( " ".join( meaningful_words ))  
import re

import nltk

from nltk.corpus import stopwords

from nltk.stem import WordNetLemmatizer

wordnet_lemmatizer = WordNetLemmatizer()
num_reviews = training_data['review'].size

print("Cleaning and parsing the training set movie reviews...\n")

clean_train_reviews = []

for i in range( 0, num_reviews ):

    # If the index is evenly divisible by 1000, print a message

    if( (i+1)%1000 == 0 ):

        print ("Review %d of %d\n" % ( i+1, num_reviews ))                                                                    

    clean_train_reviews.append( review_to_words( training_data["review"][i] ))

clean_train_reviews[0]
from sklearn.feature_extraction.text import CountVectorizer
# Initialize the "CountVectorizer" object, which is scikit-learn's bag of words tool.  

vectorizer = CountVectorizer(analyzer = "word",

                             tokenizer = None, 

                             preprocessor = None,

                             stop_words = None,  

                             max_features = 5000)
train_data_features = vectorizer.fit_transform(clean_train_reviews)
# Numpy arrays are easy to work with, so convert the result to an array

train_data_features = train_data_features.toarray()
#25000 reviews and 5000 unique words(5000 was the number we set in max_features attribute in the CountVectorizer function

#to limit the size of dictionary of unique words to 5000)

train_data_features.shape
# Take a look at the words in the vocabulary

vocab = vectorizer.get_feature_names()

print(vocab)
#Using RandomForest Classifier to classify reviews

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(train_data_features, training_data["sentiment"], 

                                                    test_size=0.2)
RF = RandomForestClassifier(n_estimators = 100)

RF.fit( X_train, y_train )
predictions = RF.predict(X_test)
from sklearn.metrics import confusion_matrix,classification_report
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))
test_data = pd.read_csv('../input/word2vec-nlp-tutorial/testData.tsv',header=0,delimiter='\t',quoting=3)
test_data.head()
num_reviews = len(test_data["review"])

clean_test_reviews = [] 
print("Cleaning and parsing the test set movie reviews...\n")

for i in range(0,num_reviews):

    if( (i+1) % 1000 == 0 ):

        print("Review %d of %d\n" % (i+1, num_reviews))

    clean_review = review_to_words( test_data["review"][i] )

    clean_test_reviews.append( clean_review )



# Get a bag of words for the test set, and convert to a numpy array

test_data_features = vectorizer.transform(clean_test_reviews)

test_data_features = test_data_features.toarray()
# Use the random forest to make sentiment label predictions

result = RF.predict(test_data_features)



# Copy the results to a pandas dataframe with an "id" column and

# a "sentiment" column

output = pd.DataFrame( data={"id":test_data["id"], "sentiment":result} )



# Use pandas to write the comma-separated output file

output.to_csv( "Bag_of_Words_model.csv", index=False, quoting=3 )

import tensorflow as tf
from tensorflow.python.keras.models import Sequential

from tensorflow.python.keras.layers import Dense

from tensorflow.python import keras
#building the model and compiling it

model = keras.Sequential([

    keras.layers.Dense(128,input_shape=(5000,),activation=tf.nn.relu),

    keras.layers.Dense(128,activation=tf.nn.relu),

    keras.layers.Dense(1, activation=tf.nn.sigmoid)

])



model.compile(optimizer='adam', 

              loss='binary_crossentropy',

              metrics=['accuracy'])
model.fit(X_train,y_train,epochs=10)
predictions_ANN = model.predict(X_test)
#to convert the output into binary True False form so that I can be compared with y_test

predictions_ANN = (predictions_ANN>0.9)
predictions_ANN
print(confusion_matrix(y_test,predictions_ANN))
print(classification_report(y_test,predictions_ANN))
result_ANN = model.predict(test_data_features)

result_ANN = result_ANN>0.9
result_ANN
result_ANN = [1 if res==True else 0 for res in result_ANN]
result_ANN
# Copy the results to a pandas dataframe with an "id" column and

# a "sentiment" column

output = pd.DataFrame( data={"id":test_data["id"], "sentiment":result_ANN} )



# Use pandas to write the comma-separated output file

output.to_csv( "Bag_of_Words_model_ANN.csv", index=False, quoting=3 )

unlabelled_data = pd.read_csv('../input/word2vec-nlp-tutorial/unlabeledTrainData.tsv',delimiter='\t',quoting=3,header=0)
unlabelled_data.head()
unlabelled_data['review'][0]
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
# Download the punkt tokenizer for sentence splitting

import nltk.data

import nltk

nltk.download('punkt')



from nltk import word_tokenize,sent_tokenize



# Load the punkt tokenizer

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')



# Define a function to split a review into parsed sentences

def review_to_sentences( review, tokenizer, remove_stopwords=False ):

    # Function to split a review into parsed sentences. Returns a list of sentences, where each sentence is a list of words

    

    # 1. Use the NLTK tokenizer to split the paragraph into sentences

    raw_sentences = tokenizer.tokenize(review.strip())

    

    # 2. Loop over each sentence

    sentences = []

    for raw_sentence in raw_sentences:

        # If a sentence is empty, skip it

        if len(raw_sentence) > 0:

            # Otherwise, call review_to_wordlist to get a list of words

            sentences.append( review_to_wordlist( raw_sentence, \

              remove_stopwords ))

    #

    # Return the list of sentences (each sentence is a list of words,

    # so this returns a list of lists

    return sentences
sentences = []  # Initialize an empty list of sentences



print("Parsing sentences from training set")

for review in training_data["review"]:

    sentences += review_to_sentences(review, tokenizer)



print("Parsing sentences from unlabeled set")

for review in unlabelled_data["review"]:

    sentences += review_to_sentences(review, tokenizer)

# Import the built-in logging module and configure it so that Word2Vec creates nice output messages

import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',level=logging.INFO)



# Set values for various parameters

num_features = 300    # Word vector dimensionality                      

min_word_count = 40   # Minimum word count                        

num_workers = 4       # Number of threads to run in parallel

context = 10          # Context window size                                                                                    

downsampling = 1e-3   # Downsample setting for frequent words



# Initialize and train the model (this will take some time)

from gensim.models import word2vec

print("Training model...")

model = word2vec.Word2Vec(sentences, workers=num_workers, \

            size=num_features, min_count = min_word_count, \

            window = context, sample = downsampling)



# If you don't plan to train the model any further, calling 

# init_sims will make the model much more memory-efficient.

model.init_sims(replace=True)



# It can be helpful to create a meaningful model name and 

# save the model for later use. You can load it later using Word2Vec.load()

model_name = "300features_40minwords_10context"

model.save(model_name)
model.wv.doesnt_match("man woman child kitchen".split())
model.wv.doesnt_match(['boy','girl','sun','child'])
model.wv.most_similar("man")
model.wv.most_similar("queen")
model.wv.most_similar("awful")
model.wv.most_similar("great")
model.wv.most_similar('dog')
# Load the model that we created

from gensim.models import Word2Vec

model = Word2Vec.load("300features_40minwords_10context")
type(model.wv.syn0)
model.wv.syn0.shape
model['great']
def makeFeatureVec(words, model, num_features):

    # Function to average all of the word vectors in a given paragraph

    

    # Pre-initialize an empty numpy array (for speed)

    featureVec = np.zeros((num_features,),dtype="float32")

    

    nwords = 0.

     

    # Index2word is a list that contains the names of the words in 

    # the model's vocabulary. Convert it to a set, for speed 

    index2word_set = set(model.wv.index2word)

    #

    # Loop over each word in the review and, if it is in the model's

    # vocaublary, add its feature vector to the total

    for word in words:

        if word in index2word_set: 

            nwords = nwords + 1.

            #here we are adding the feature vectors of all the words that were in the review

            featureVec = np.add(featureVec,model[word])

    # 

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

        # Print a status message every 1000th review

        if counter%1000 == 0:

            print("Review %d of %d" % (counter, len(reviews)))

        # Call the function (defined above) that makes average feature vectors

        reviewFeatureVecs[counter] = makeFeatureVec(review, model,num_features)

        # Increment the counter

        counter = counter + 1

    return reviewFeatureVecs
# ****************************************************************

# Calculate average feature vectors for training and testing sets,

# using the functions we defined above. Notice that we now use stop word

# removal.



clean_train_reviews = []

for review in training_data["review"]:

    clean_train_reviews.append( review_to_wordlist( review,remove_stopwords=True ))



trainDataVecs = getAvgFeatureVecs( clean_train_reviews, model, num_features )



print ("Creating average feature vecs for test reviews")

clean_test_reviews = []

for review in test_data["review"]:

    clean_test_reviews.append( review_to_wordlist( review,remove_stopwords=True ))



testDataVecs = getAvgFeatureVecs( clean_test_reviews, model, num_features )
X_train_vec,X_test_vec,y_train_vec,y_test_vec = train_test_split(trainDataVecs,training_data['sentiment'])
from sklearn.ensemble import RandomForestClassifier

forest_train = RandomForestClassifier( n_estimators = 100 )



print ("Fitting a random forest to labeled training data...")

forest_train.fit( X_train_vec, y_train_vec )



# Test & extract results 

result = forest_train.predict( X_test_vec )
print(confusion_matrix(y_test_vec,result))

print('\n')

print(classification_report(y_test_vec,result))
# Fit a random forest to the training data, using 100 trees

forest = RandomForestClassifier( n_estimators = 100 )



forest.fit( trainDataVecs, training_data["sentiment"] )



# Test & extract results 

result_to_be_submitted = forest.predict( testDataVecs )



# Write the test results 

output = pd.DataFrame( data={"id":test_data["id"], "sentiment":result_to_be_submitted} )

output.to_csv( "Word2Vec_AverageVectors.csv", index=False, quoting=3 )

from sklearn.cluster import KMeans

import time



start = time.time() # Start time



# Set "k" (num_clusters) to be 1/5th of the vocabulary size, or an

# average of 5 words per cluster

word_vectors = model.wv.syn0

num_clusters = int(word_vectors.shape[0] / 5)



# Initalize a k-means object and use it to extract centroids

kmeans_clustering = KMeans( n_clusters = num_clusters )

idx = kmeans_clustering.fit_predict( word_vectors )



# Get the end time and print how long the process took

end = time.time()

elapsed = end - start

print ("Time taken for K Means clustering: ", elapsed, "seconds.")
# Create a Word / Index dictionary, mapping each vocabulary word to a cluster label                                                                                            

word_centroid_map = dict(zip( model.wv.index2word, idx ))
values_word_centroid = list(word_centroid_map.values())
values_word_centroid[0]
keys_word_centroid = list(word_centroid_map.keys())
# For the first 10 clusters

for cluster in range(0,10):

    #

    # Print the cluster number  

    print ("\nCluster %d" % cluster)

    #

    # Find all of the words for that cluster number, and print them out

    words = []

    for i in range(0,len(word_centroid_map.values())):

        if( values_word_centroid[i] == cluster ):

            words.append(keys_word_centroid[i])

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

train_centroids = np.zeros( (training_data["review"].size, num_clusters), \

    dtype="float32" )



# Transform the training set reviews into bags of centroids

counter = 0

for review in clean_train_reviews:

    train_centroids[counter] = create_bag_of_centroids( review, \

        word_centroid_map )

    counter += 1



# Repeat for test reviews 

test_centroids = np.zeros(( test_data["review"].size, num_clusters), \

    dtype="float32" )



counter = 0

for review in clean_test_reviews:

    test_centroids[counter] = create_bag_of_centroids( review, \

        word_centroid_map )

    counter += 1
# Fit a random forest and extract predictions 

forest = RandomForestClassifier(n_estimators = 100)



# Fitting the forest may take a few minutes

print ("Fitting a random forest to labeled training data...")

forest = forest.fit(train_centroids,training_data["sentiment"])

result = forest.predict(test_centroids)



# Write the test results 

output = pd.DataFrame(data={"id":test_data["id"], "sentiment":result})

output.to_csv( "BagOfCentroids.csv", index=False, quoting=3 )