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
dataset_filename = os.listdir("../input")[0]

dataset_path = os.path.join("..","input",'labeledTrainData.tsv')

print("Open file:", dataset_path)
import pandas as pd       

train = pd.read_csv(dataset_path, header=0, \

                    delimiter="\t", quoting=3)
train.shape
train.columns.values
train["review"][0]
from bs4 import BeautifulSoup
example1 = BeautifulSoup(train["review"][0])

example1.get_text()
# Not a reliable practise to remove markups using regular expressions
import re

letters_only = re.sub("[0-9]+", " ", example1.get_text())

print(letters_only)

lower_case  = letters_only.lower()

words = lower_case.split()
import nltk

from nltk.corpus import stopwords

stop = stopwords.words("english")
def review_to_words(raw_review ):

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

    sto = set(stopwords.words("english"))                  

    # 

    # 5. Remove stop words

    meaningful_words = [w for w in words if not w in stop]   

    #

    # 6. Join the words back into one string separated by space, 

    # and return the result.

    return( " ".join( meaningful_words ))   
num_reviews = train["review"].size

clean_train_reviews = []



# Loop over each review; create an index i that goes from 0 to the length

# of the movie review list 

for i in range(0, num_reviews):

    if( (i+1)%10000 == 0 ):

        print ("Review %d of %d\n" % ( i+1, num_reviews ))

    clean_train_reviews.append(review_to_words(

        train["review"][i]))
#Bag of words model

print ("Creating the bag of words...\n")

from sklearn.feature_extraction.text import CountVectorizer



#initialize countvectorizer

cv = CountVectorizer(analyzer = "word",

                             tokenizer = None, 

                             preprocessor = None,

                             stop_words = None,

                             max_features = 5000)

# fit_transform() does two functions: First, it fits the model

# and learns the vocabulary; second, it transforms our training data

# into feature vectors. The input to fit_transform should be a list of 

# strings.

train_data_features = cv.fit_transform(clean_train_reviews)

#train_data_features is a sparse matrix

train_data_features = train_data_features.toarray()
vocab = cv.get_feature_names()
import numpy as np



#Sum up the counts of each vocablury word

dist = np.sum(train_data_features, axis=0)



# For each, print the vocabulary word and the number of times it 

# appears in the training set

for tag, count in zip(vocab, dist):

    print(count, tag)
print ("Training the random forest...")

from sklearn.ensemble import RandomForestClassifier



# Initialize a Random Forest classifier with 100 trees

forest = RandomForestClassifier(n_estimators = 100) 



forest = forest.fit(train_data_features, train['sentiment'])
dataset_filename = os.listdir("../input")[0]

dataset_path = os.path.join("..","input",'testData.tsv')

print("Open file:", dataset_path)
# Read the test data

test = pd.read_csv(dataset_path, header=0, delimiter="\t", \

                   quoting=3 )



# Verify that there are 25,000 rows and 2 columns

print (test.shape)



# Create an empty list and append the clean reviews one by one

num_reviews = len(test["review"])

clean_test_reviews = [] 



print ("Cleaning and parsing the test set movie reviews...\n")

for i in range(0,num_reviews):

    if( (i+1) % 1000 == 0 ):

        print ("Review %d of %d\n" % (i+1, num_reviews))

    clean_review = review_to_words( test["review"][i] )

    clean_test_reviews.append( clean_review )



# Get a bag of words for the test set, and convert to a numpy array

test_data_features = cv.transform(clean_test_reviews)

test_data_features = test_data_features.toarray()



# Use the random forest to make sentiment label predictions

result = forest.predict(test_data_features)



# Copy the results to a pandas dataframe with an "id" column and

# a "sentiment" column

output = pd.DataFrame( data={"id":test["id"], "sentiment":result} )



# Use pandas to write the comma-separated output file

output.to_csv( "Bag_of_Words_model.csv", index=False, quoting=3 )