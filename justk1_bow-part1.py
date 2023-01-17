# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from bs4 import BeautifulSoup  

import re

import nltk

from nltk.corpus import stopwords



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/labeledTrainData.tsv', header=0,delimiter="\t",quoting=3)
train.head()
print ('Number of lines in training data are',train.shape[0])
print (train['review'][0])





# Note that this text contains html
def review_to_words(raw_review):

    # Remove HTML tags and markup with Beautiful soup

    review_text = BeautifulSoup(raw_review,'lxml').get_text()

    

    # Remove numbers and punctuations

    letters_text = re.sub('[^a-zA-Z]',

                       ' ',

                       review_text)

    

    # Change words to lower case and split to individual words

    words = letters_text.lower().split()

    

    # Remove stopwords

    stops = set(stopwords.words("english"))  

    meaningful_words = [w for w in words if not w in stops]

    

    return ( ' '.join(meaningful_words))
num_reviews = train.review.size

clean_train_reviews = []
for r in range(num_reviews):

    clean_train_reviews.append(review_to_words(train['review'][r]))
clean_train_reviews[0]
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(analyzer='word',max_features = 5000)

train_data_features = vectorizer.fit_transform(clean_train_reviews)

train_data_features = train_data_features.toarray()
train_data_features.shape
vocab = vectorizer.get_feature_names()

print (vocab)
import numpy as np



# Sum up the counts of each vocabulary word

dist = np.sum(train_data_features, axis=0)



# For each, print the vocabulary word and the number of times it 

# appears in the training set

for tag, count in zip(vocab, dist):

    print (count, tag)
from sklearn.ensemble import RandomForestClassifier

forest = RandomForestClassifier(n_estimators = 120) 

forest = forest.fit( train_data_features, train["sentiment"] )
test = pd.read_csv("../input/testData.tsv", header=0, delimiter="\t", \

                   quoting=3 )



print (test.shape)



num_reviews = len(test["review"])

clean_test_reviews = [] 



for i in range(num_reviews):

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



# Use pandas to write the comma-separated output file

output.to_csv( "Bag_of_Words_model.csv", index=False, quoting=3 )