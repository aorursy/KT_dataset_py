import numpy as np

import pandas as pd



from bs4 import BeautifulSoup

import re

import nltk

from nltk.corpus import stopwords

from tqdm import tqdm



from sklearn.feature_extraction.text import CountVectorizer

from sklearn.ensemble import RandomForestClassifier



import os

import warnings

warnings.filterwarnings("ignore")

import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)



# Method 2 Libraries

from gensim.models.word2vec import Word2Vec

from sklearn.cluster import KMeans

import time



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# Load Data

train = pd.read_csv('../input/word2vec-nlp-tutorial/labeledTrainData.tsv.zip', quoting=3, delimiter='\t')

test = pd.read_csv('../input/word2vec-nlp-tutorial/testData.tsv.zip', quoting=3, delimiter='\t')

print('Train Shape:', train.shape)

print('Test Shape:', test.shape)

train.head()
# Clean Data

def review_to_words(raw_review):

    review_text = BeautifulSoup(raw_review).get_text()           # remove HTML

    letters_only = re.sub('[^a-zA-Z]', ' ', review_text)          # remove digits

    words = letters_only.lower().split()                         # lower and split into array 

    stops = set(stopwords.words('english'))                      # get stops, transform into set for speed 

    meaningful_words = [w for w in words if w not in stops]      # get meaningful words without stops

    return ' '.join(meaningful_words)                            # join meaningful words together



num_reviews = train.shape[0]

clean_train_reviews = []

clean_test_reviews = []

for i in tqdm(range(num_reviews)):

    clean_train_reviews.append(review_to_words(train['review'][i]))

    clean_test_reviews.append(review_to_words(test['review'][i]))
# Create Bag of Words

vectorizer = CountVectorizer(max_features=5000)                        # initialize CountVectorizer

train_data_features = vectorizer.fit_transform(clean_train_reviews)    # learns vocab & transforms words into feature vectors

train_data_features = train_data_features.toarray()                    # convert to array for ease of use



test_data_features = vectorizer.transform(clean_test_reviews)          # test data

test_data_features = test_data_features.toarray()



vocab = vectorizer.get_feature_names()

print('Train Features Shape:', train_data_features.shape)

print('Test Features Shape:', test_data_features.shape)
# Train RandomForestClassifier and Predict

forest = RandomForestClassifier(n_estimators = 100)           # initialize RandomForestClassifier with 100 trees

forest = forest.fit(train_data_features, train['sentiment'])  # fit forest

result = forest.predict(test_data_features)                   # predict

output = pd.DataFrame(data={'id': test['id'], 'sentiment': result})

output.to_csv('Bag_of_Words_model.csv', index=False, quoting=3)
# Load Data

labeled_train = pd.read_csv('../input/word2vec-nlp-tutorial/labeledTrainData.tsv.zip', quoting=3, delimiter='\t')

unlabeled_train = pd.read_csv('../input/word2vec-nlp-tutorial/unlabeledTrainData.tsv.zip', quoting=3, delimiter='\t')

test = pd.read_csv('../input/word2vec-nlp-tutorial/testData.tsv.zip', quoting=3, delimiter='\t')
# Preprocess Data

def review_to_wordlist(review, remove_stopwords=False, remove_numbers=False):   # keep stop words & numbers for Word2vec

    review_text = BeautifulSoup(review).get_text()

    if remove_numbers:

        review_text = re.sub("[^a-zA-Z]"," ", review_text)

    else:

        review_text = re.sub("[^a-zA-Z0-9]"," ", review_text)

    words = review_text.lower().split()

    if remove_stopwords:

        stops = set(stopwords.words("english"))

        words = [w for w in words if not w in stops]

    return(words)



# Word2Vec expects a list of sentences, where each sentence is a list of words

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')            # load punkt tokenizer



def review_to_sentences(review, tokenizer, remove_stopwords=False):

    raw_sentences = tokenizer.tokenize(review.strip())     # get list of sentences

    sentences = []

    for raw_sentence in raw_sentences:

        if len(raw_sentence) > 0:                          # get list of words 

            sentences.append(review_to_wordlist(raw_sentence, remove_stopwords))

    return sentences



sentences = []

for review in labeled_train['review']:                         # labeled train set

    sentences += review_to_sentences(review, tokenizer)

for review in unlabeled_train['review']:                       # unlabeled train set

    sentences += review_to_sentences(review, tokenizer)
# Initialize and Train Model

num_features = 300    # Word vector dimensionality                      

min_word_count = 40   # Minimum word count                        

num_workers = 4       # Number of threads to run in parallel

context = 10          # Context window size                                                                                    

downsampling = 1e-3   # Downsample setting for frequent words



model = Word2Vec(sentences, workers=num_workers, size=num_features, 

                          min_count=min_word_count, window=context, sample=downsampling)



# init_sims works only when the model will not be trained further, and is more memory-efficient.

model.init_sims(replace=True)



# Save Model

model_name = "300features_40minwords_10context"

model.save(model_name)



# Exploring Results

print('Most different in france, england, germany, berlin:', model.doesnt_match("france england germany berlin".split()))

print('Most similar to awful:', model.most_similar("awful"))
# Use Model to Predict: Vector Averaging



# Take word vectors and transform them into a feature set that is the same length for every review.

# To combine the words in each review, we can average the word vectors in a review (removed stop words to remove extra noise)



def makeFeatureVec(words, model, num_features):

    """averages all word vectors in a paragraph"""

    featureVec = np.zeros((num_features,), dtype="float32")

    nwords = 0

    index2word_set = set(model.wv.index2word)           # a set of words in the model's vocabulary

    for word in words:                               # if a word is in the vocaublary, add its feature vector to the total

        if word in index2word_set: 

            nwords = nwords + 1

            featureVec = np.add(featureVec, model[word])

    featureVec = np.divide(featureVec,nwords)

    return featureVec



def getAvgFeatureVecs(reviews, model, num_features):

    # calculate the average feature vector for n reviews and return a 2D numpy array 

    counter = 0

    reviewFeatureVecs = np.zeros((len(reviews), num_features), dtype="float32")

    for review in tqdm(reviews):

       reviewFeatureVecs[counter] = makeFeatureVec(review, model, num_features)

       counter = counter + 1

    return reviewFeatureVecs





# Calculate average feature vectors for training and testing sets, using stop word removal.

clean_train_reviews = []

for review in train["review"]:

    clean_train_reviews.append(review_to_wordlist(review, remove_stopwords=True))

trainDataVecs = getAvgFeatureVecs(clean_train_reviews, model, num_features)



clean_test_reviews = []

for review in test["review"]:

    clean_test_reviews.append(review_to_wordlist(review, remove_stopwords=True))

testDataVecs = getAvgFeatureVecs(clean_test_reviews, model, num_features)





# Use average paragraph vectors to train a random forest

forest = RandomForestClassifier(n_estimators = 100)   # fit RandomForest of 100 trees

forest = forest.fit(trainDataVecs, train["sentiment"])

result = forest.predict(testDataVecs)                   # predict



output = pd.DataFrame(data={"id":test["id"], "sentiment":result})

output.to_csv("Word2Vec_AverageVectors.csv", index=False, quoting=3)