import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline
training_data = pd.read_csv('../input/word2vec-nlp-tutorial/labeledTrainData.tsv.zip',header=0,delimiter='\t',quoting=3)
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
test_data = pd.read_csv('../input/word2vec-nlp-tutorial/testData.tsv.zip',header=0,delimiter='\t',quoting=3)
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

output.to_csv( "submission.csv", index=False, quoting=3 )
