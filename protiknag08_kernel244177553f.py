import numpy as np

import pandas as pd

from bs4 import BeautifulSoup

import re



import nltk

from nltk.stem import PorterStemmer

from nltk.stem import LancasterStemmer

from nltk.corpus import stopwords



from sklearn.feature_extraction.text import CountVectorizer
train = pd.read_csv('../input/word2vec-nlp-tutorial/labeledTrainData.tsv', quoting=3, delimiter='\t', header=0)

test = pd.read_csv('../input/word2vec-nlp-tutorial/testData.tsv', quoting=3, delimiter='\t', header=0)

submission = pd.read_csv('../input/word2vec-nlp-tutorial/sampleSubmission.csv', quoting=3, delimiter='\t', header=0)

unlabeledTrain = pd.read_csv('../input/word2vec-nlp-tutorial/unlabeledTrainData.tsv', quoting=3, delimiter='\t', header=0)
print("Train Shape = ", train.shape)

print("Test Shape = ", test.shape)
print(train.columns.values)

print(test.columns.values)
stemmer = PorterStemmer()
def review_to_words(raw_review):

    review_text = BeautifulSoup(raw_review,'lxml').get_text()       

    letters_only = re.sub("[^a-zA-Z]", " ", review_text) 

    letters_only = re.sub("!", "WOW ", letters_only)

    words = letters_only.lower().split()                 

    stops = set(stopwords.words("english"))              

    meaningful_words = [w for w in words if not w in stops]

    singles = [stemmer.stem(x) for x in meaningful_words]

    return( " ".join(singles)) 
clean_train_reviews = []

num_reviews = len(train['review'])

for i in range( 0, num_reviews ):

    if( (i+1)%1000 == 0 ):

        print("Review %d of %d\n" % ( i+1, num_reviews ))                                                                    

    clean_train_reviews.append( review_to_words( train["review"][i] ))
vectorizer = CountVectorizer(analyzer = "word", tokenizer = None, preprocessor = None, stop_words = "english", max_features = 7000) 

train_data_features = vectorizer.fit_transform(clean_train_reviews)



train_data_features = train_data_features.toarray()
print(train_data_features.shape)
vocab = vectorizer.get_feature_names()

print(vocab)
dist = np.sum(train_data_features, axis=0)



for tag, count in zip(vocab, dist):

    print(count, tag)
print("Training the random forest...")

from sklearn.ensemble import RandomForestClassifier



forest = RandomForestClassifier(n_estimators = 300) 

forest = forest.fit( train_data_features, train["sentiment"] )
test = pd.read_csv("../input/word2vec-nlp-tutorial/testData.tsv", header=0, delimiter="\t", \

                   quoting=3 )



print(test.shape)



num_reviews = len(test["review"])

clean_test_reviews = [] 



print("Cleaning and parsing the test set movie reviews...\n")

for i in range(0,num_reviews):

    if( (i+1) % 5000 == 0 ):

        print("Review %d of %d\n" % (i+1, num_reviews))

    clean_review = review_to_words( test["review"][i] )

    clean_test_reviews.append( clean_review )



test_data_features = vectorizer.transform(clean_test_reviews)

test_data_features = test_data_features.toarray()



result = forest.predict(test_data_features)



output = pd.DataFrame( data={"id":test["id"], "sentiment":result} )

output.to_csv( "Bag_of_Words_model.csv", index=False, quoting=3 )