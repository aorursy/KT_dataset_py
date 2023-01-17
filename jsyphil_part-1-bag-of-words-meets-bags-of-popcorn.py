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
from bs4 import BeautifulSoup as bs



import nltk



from sklearn.feature_extraction.text import CountVectorizer

from sklearn.ensemble import RandomForestClassifier



import re
# Import the pandas package, then use the "read_csv" function to read

# the labeled training data

train = pd.read_csv("../input/word2vec-nlp-tutorial/labeledTrainData.tsv.zip", header=0, \

                    delimiter="\t", quoting=3)
train.shape
train.columns.values
print(train['review'][0])
example1 = bs(train['review'][0])

print(example1.get_text())
# Use regular expressions to do a find-and-replace

letters_only = re.sub("[^a-zA-Z]",           # The pattern to search for

                      " ",                   # The pattern to replace it with

                      example1.get_text() )  # The text to search

print(letters_only)
lower_case = letters_only.lower()        # Convert to lower case

words = lower_case.split()               # Split into words
#nltk.download()
from nltk.corpus import stopwords # Import the stop word list

print(stopwords.words("english"))
# Remove stop words from "words"

words = [w for w in words if not w in stopwords.words("english")]

print(words)
def review_to_words(raw_review):

    '''

    Function to convert a raw review to a string of words

    The input is a single string (a raw movie review), and 

    the output is a single string (a preprocessed movie review)

    '''

    

    # 1. Remove HTML

    review_text = bs(raw_review).get_text() 

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

    #

    # 6. Join the words back into one string separated by space, 

    # and return the result.

    return( " ".join( meaningful_words ))  
clean_review = review_to_words( train["review"][0] )

print(clean_review)
# Get the number of reviews based on the dataframe column size

num_reviews = train["review"].size



# Initialize an empty list to hold the clean reviews

clean_train_reviews = []



# Loop over each review; create an index i that goes from 0 to the length

# of the movie review list 



print("Cleaning and parsing the training set movie reviews...\n")

for i in range( 0, num_reviews ):

    # If the index is evenly divisible by 1000, print a message

    if((i+1)%1000 == 0):

        print(f"Review {i+1} of {num_reviews}\n")                                                                    

    clean_train_reviews.append(review_to_words(train["review"][i]))

print("Finished cleaning and parsing the training set movie reviews...\n")
# Initialize the "CountVectorizer" object, which is scikit-learn's

# bag of words tool.  

vectorizer = CountVectorizer(analyzer = "word",   \

                             tokenizer = None,    \

                             preprocessor = None, \

                             stop_words = None,   \

                             max_features = 5000) 



# fit_transform() does two functions: First, it fits the model

# and learns the vocabulary; second, it transforms our training data

# into feature vectors. The input to fit_transform should be a list of 

# strings.

print("Creating the bag of words...\n")

train_data_features = vectorizer.fit_transform(clean_train_reviews)
print("Shape of Bag of Words: ", train_data_features.shape)

print("Example review: ",clean_train_reviews[5])

print("Related Bag of Words:\n",train_data_features[5])
# Numpy arrays are easy to work with, so convert the result to an 

# array

train_data_features_array = train_data_features.toarray()
# Take a look at the words in the vocabulary

vocab = vectorizer.get_feature_names()

print(vocab)
print(vocab[2912])
# Sum up the counts of each vocabulary word

dist = np.sum(train_data_features, axis=0)



# For each, print the vocabulary word and the number of times it 

# appears in the training set

for tag, count in zip(vocab, dist):

    print(count, tag)
print("Training the random forest...")



# Initialize a Random Forest classifier with 100 trees

forest = RandomForestClassifier(n_estimators = 100) 



# Fit the forest to the training set, using the bag of words as 

# features and the sentiment labels as the response variable

#

# This may take a few minutes to run

forest = forest.fit( train_data_features, train["sentiment"] )

print("Finished training the random forest...")
# Read the test data

test = pd.read_csv("../input/word2vec-nlp-tutorial/testData.tsv.zip",

                   header=0,

                   delimiter="\t",

                   quoting=3)



# Verify that there are 25,000 rows and 2 columns

print(test.shape)
# Create an empty list and append the clean reviews one by one

num_reviews = len(test["review"])

clean_test_reviews = [] 



print("Cleaning and parsing the test set movie reviews...\n")

for i in range(0,num_reviews):

    if( (i+1) % 1000 == 0 ):

        print("Review %d of %d\n" % (i+1, num_reviews))

    clean_review = review_to_words( test["review"][i] )

    clean_test_reviews.append( clean_review )



# Get a bag of words for the test set, and convert to a numpy array

test_data_features = vectorizer.transform(clean_test_reviews)

test_data_features = test_data_features.toarray()



# Use the random forest to make sentiment label predictions

result = forest.predict(test_data_features)



# Copy the results to a pandas dataframe with an "id" column and

# a "sentiment" column

output = pd.DataFrame( data={"id":test["id"], "sentiment":result} )

print("Finished cleaning and parsing the test set movie reviews...\n")
# Use pandas to write the comma-separated output file

output.to_csv( "Bag_of_Words_model.csv", index=False, quoting=3 )