## Lets begin with Project - IMDB movie review - Sentiment Analysis Project using python 3.7
#  and Jupiter IDE.
# Loading Libraries and Data


# Import important libraries to work in Python
 
import pandas as pd               # for pandas in Python
import numpy as np                # for numpy in Python

from bs4 import BeautifulSoup     # BeautifulSoup is used to remove html tags from the text

import re                         # For regular expressions

from nltk.corpus import stopwords #  to remove stopwords in the movie review

# After importing the important libraries it's time to read the Imdb Movie Review Project
# It has train data and test data.

# By default in Python gives you 5 row of the data when you read the data into it followed by train.head()
# Suppose if you want to read 10 rows, train.head(10) will do that.

train = pd.read_csv('../input/labeledTrainData.tsv',header = 0, delimiter = '\t')
train.head(5)

## EDA 

## Now, to look into the data variables, type the code as shown below. 

train.columns.values

## It gives an array of the variables in the data set.
## To know about the dimension of data:
train.shape

## This gives an idea about number of rows and columns, i.e shape of te data set
## Below, you can see that we have over 25,000 observations in the train set
## To see the 1st movie review which is stored in the zero index in python, 

print (train["review"][0])

# Please note that the raw review has punctuations, small capse letters, capital letters...etc

## Now we require to clean the data, i.e data cleansing has to be done.
## It's time to import BeautifulSoup into your workspace

## what is BeautifulSoup and why it is important?

# Beautiful Soup is a library that makes it easy to scrape information from web pages. 
# It sits atop an HTML or XML parser, providing Pythonic idioms for iterating, searching, 
# and modifying the parse tree.

## you can read about b54 further here, https://pypi.org/project/beautifulsoup4/

from bs4 import BeautifulSoup 

# Let's start the BeautifulSoup object on a single movie review, later on let's define a function which will be helpful 
# to check for 25k rows at once, instead of repeating for 25k rows.

example1 = BeautifulSoup(train["review"][0], "lxml")

# Print the raw review and then the output of get_text(),
# To compare data before and after, let's print both.

print ('\nNotice Please: \n')
print ('\nImdb Movie Raw Review - Before BeautifulSoup treatment on Train data \n')
print (train["review"][0])

print ('\nImdb Movie Review - After BeautifulSoup treatment on Train data \n')
print (example1.get_text())

# This Project deals with sentiment analysis of movie reviews, and in real sense, it is possible 
## that even "!!!" or ":-(" or ":))", could carry sentiments/expressions, which makes sense, 
## and they should be treated as words even during analysis.

## However, for simplicity let's remove the punctuations. 
## Once we get the desired result, let's see how to play around the data set with punctuations and compare 
## the model's performance with and without expressions. We need to explore that way.

## To remove punctuation and numbers, in Python, a package called re, for dealing with regular 
#  expressions is widely used. 
#  To know more about re, follow -> https://docs.python.org/3/library/re.html

## let' import the re package from library

import re

letters_only = re.sub("[^a-zA-Z]", " ",example1.get_text() )
print (letters_only)
## At this satge we need to have one more important package called, NLTK
# NLTK is a Natural Language Tool Kit, that helps with the entire natural language processing methodology

# Check whether it is installed in your systme.
# Try with import nltk in Jupyter, if you get an error as no module named/defined nltk, then You have to download it.
## Enter in commond prompt pip install nltk, if you find error then check in jupyter

import nltk

# nltk.download()

## New window will open - NLTK Downloader
## select popular and tests and start download, just wait for some time, as 
## unzipping process from server folders takes some time. 
## Once you see "finished downloading collection all" its over, you can start. 

## Notice: After the download process is over, remove nltk.download written below import nltk, or make it a tag, 
# otherwise it will stop processing when the kernel -> Restarts and Run all.

## Tokenizing raw text data is an important data pre-processing steps for many NLP methods, it is 
## a process of breaking a stream of text like what we saw movie review, into words, meaningful phrases, 
## and symbols, called tokens. This is the way Python works with human language data. So NLTK module is very imp.

## Steps followed, 
## TOKENIZATION -> Convert to lower case -> Split into words -> see the words

## Let's convert IMDB movie reviews to lower case and split them into 
## individual words (called "tokenization" in NLP lingo)

# Convert to lower case
lower_case = letters_only.lower()  

# Split into words
words = lower_case.split()  
words
## The words list has many words which are very common/frequent in English language and they need to be 
## removed from the data, they are called stop words in english.

## Let's import a stop word list from the Python Natural Language Toolkit (NLTK), including stop words

# To view the list of English-language stop words
# Import the stop word list to see how it would look like.

from nltk.corpus import stopwords 
print (stopwords.words("english"))
# Now, we need to to Remove the above seen stop words from our movie review data.

words = [w for w in words if not w in stopwords.words("english")]
print ('\nAfter Removing Stopwords from Movie Review Data: Review stored in the zero Index looks as below\n')
print (words)
## Notice, now we have code to clean individual review separately, it's just a sample.
## but in our data we need to clean 25,000 such reviews! same with training and test data. 
## To make our code reusable, let's create a function that can be called many times instaed of doing it over and again.
def review_to_words( raw_review ):
    # Function to convert a raw review to a string of words
    # The input is a single string (a raw movie review), and 
    # the output is a single string (a preprocessed movie review)
    
    # 1. To Remove HTML in the movie review 
    review_text = BeautifulSoup(raw_review, "lxml").get_text() 
    
    # 2. To Remove non-letters in the movie review      
    letters_only = re.sub("[^a-zA-Z]", " ", review_text) 
    
    # 3. To Convert to lower case, split into individual words in the movie review 
    words = letters_only.lower().split()                             
    
    # 4. To Convert to:
    #    In Python, searching a set is much faster than searching a list,
    #    so convert the stop words to a set
    stops = set(stopwords.words("english"))                  
    
    # 5. Remove stop words in the movie review 
    meaningful_words = [w for w in words if not w in stops]   
    
    # 6. Join the words back into one string separated by space, 
    # and return the result.
    return( " ".join( meaningful_words )) 
clean_review = review_to_words(train["review"][0] )
print (clean_review)
## Now it should be noticed here that, print (clean_review) has given the same output as before.
## To read few more reviews individually,
clean_review = review_to_words(train["review"][1] )
print (clean_review)
## To read few more interesting reviews individually,
clean_review = review_to_words(train["review"][5] )
print (clean_review)
## So this process should go on for all the 25k reviews in the data set, to run "clean_review" 25000 times 
## is time consuming and not a correct way as there are 25000 reviews!!

## Instead, let's loop through and clean all of the Imdb training set at once.
# Get the number of reviews based on the dataframe column size
num_reviews = train["review"].size

# Initialize an empty list to hold the clean reviews
clean_train_reviews = []

# Loop over each review; create an index i that goes from 0 to the length
# of the movie review list 
for i in range( 0, num_reviews ):
    # Call our function for each one, and add the result to the list of
    # clean reviews
    clean_train_reviews.append( review_to_words( train["review"][i] ) )
    clean_train_reviews = []
    
print ("Please wait a while, Cleaning, parsing the training set of Imdb movie reviews is going on...\n")   
for i in range( 0, num_reviews ):
    # If the index is evenly divisible by 1000, print a message
    if( (i+1)%1000 == 0 ):
        print ("Review %d of %d\n" % ( i+1, num_reviews ))                                                                    
    clean_train_reviews.append( review_to_words( train["review"][i] ))
    
## Let's create Bag of Words out of the cleaned movie reviews data.

print ("Creating the bag of words...\n")

## To prepare cleaned text data for Machine Learning, Impoprt CountVectorizer from sklearn, which is used
## to Convert a collection of text documents (cleaned movie review) to a matrix of token counts.

from sklearn.feature_extraction.text import CountVectorizer

# Initialize the "CountVectorizer" object, which is scikit-learn's bag of words tool.  

vectorizer = CountVectorizer(analyzer = "word",   \
                             tokenizer = None,    \
                             preprocessor = None, \
                             stop_words = None,   \
                             max_features = 5000)

# Now it's time to train the model on the train data set, for this fit_transform() is used, 
# it does two functions: 
# First, it fits the model and learns the vocabulary; 
# second, it transforms our training data into feature vectors. 
# The input to fit_transform should be a list of strings.

train_data_features = vectorizer.fit_transform(clean_train_reviews)

# Numpy arrays are easy to work with, so convert the result to an array

train_data_features = train_data_features.toarray()
print (train_data_features.shape)
## NOw Bag of Words model is trained, so let's look at the words in the vocabulary of the IMDB movie review

vocab = vectorizer.get_feature_names()
print ('\n The vocabulary in the bag of words \n')
print (vocab)
## If curious! to know, the count of each word in the vocabulary:

# Sum up the counts of each vocabulary word
dist = np.sum(train_data_features, axis=0)

# For each, print the vocabulary word and the number of times it appears in the training set
print ('\n Look at here! \n')
print ('\n Count of the words in the vocabulary \n')
for tag, count in zip(vocab, dist):
   print (tag, '-', count, 'times')
## Let's begin Supervised Learning
## Let's use Random Forest classifier, which is included in scikit-learn 

## In order to reduce the time taken, let's fix number of trees to 100 for time being. 
## Later on during tuning we can increase the number of trees to make the model perform better.
print ("Training the Random Forest...........")

from sklearn.ensemble import RandomForestClassifier

forest = RandomForestClassifier(n_estimators = 100) 

# Fit the forest to the training set, using the bag of words as 
# features and the sentiment labels as the response variable
# This may take a few minutes to run
forest = forest.fit( train_data_features, train["sentiment"] )
print (forest)
## Now our model is trained with the train data set. So in order to test it's reliability call the test data
#  To Import Test Data
test = pd.read_csv('../input/testData.tsv',header = 0, delimiter = '\t')
test.head(5)
print (test.shape)
# Create an empty list and append the clean reviews one by one
num_reviews = len(test["review"])
clean_test_reviews = [] 

print ("Please wait a while, Cleaning, parsing of the test set of Imdb movie reviews is going on...\n")
for i in range(0,num_reviews):
    if( (i+1) % 1000 == 0 ):
        print ("Review %d of %d\n" % (i+1, num_reviews))
    clean_review = review_to_words( test["review"][i] )
    clean_test_reviews.append( clean_review )
# Get a bag of words for the test set, and convert to a numpy array
test_data_features = vectorizer.transform(clean_test_reviews)
test_data_features = test_data_features.toarray()
# Use the random forest to make sentiment label predictions
result = forest.predict(test_data_features)
# We need to Copy the results into a pandas dataframe with an "id" column and a "sentiment" column, 
# as this was the requirement of the project, with the original review of 0 and 1

output = pd.DataFrame( data={"id":test["id"], "sentiment":result} )

# Use pandas to write the comma-separated output file and give the name of the file 
output.to_csv( "BAG_OF_WORDS_MEETS_BAGS_OF_POPCORN.csv", index=False, quoting=3 )

