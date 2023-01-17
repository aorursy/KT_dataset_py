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
import pandas as pd

from bs4 import BeautifulSoup

import nltk

from nltk.corpus import stopwords

import re

import numpy as np

import time

from sklearn.ensemble import RandomForestClassifier

from sklearn.feature_extraction.text import CountVectorizer
train = pd.read_csv("../input/train.tsv", header = 0, delimiter = '\t')

test = pd.read_csv("../input/test.tsv", header = 0, delimiter = '\t')
train.shape
def phrase_to_words(raw_phrase):

    

    #remove any html

    phrase_text = BeautifulSoup(raw_phrase).get_text()

    

    #remove non letters

    letters = re.sub("[^A-Za-z]", " ", phrase_text)

    

    #to lowercase

    lower_letters = letters.lower().split()

    

    #remove stopwords

    stop = set(stopwords.words('english'))

    meaningful_words = [word for word in lower_letters if word not in stop]

    

    return (" ".join(meaningful_words))
#First the train set

num_phrase = train['Phrase'].size

clean_train_phrase = []

for i in range(0, num_phrase):

    if( (i+1)%10000 == 0 ):

        print ("Review %d of %d\n" % ( i+1, num_phrase ))

    clean_train_phrase.append(phrase_to_words(train['Phrase'][i]))
print("Making Bag of words")

vectorizer = CountVectorizer(analyzer = "word",   \

                             tokenizer = None,    \

                             preprocessor = None, \

                             stop_words = None,   \

                             max_features = 5000) 



train_data_features = vectorizer.fit_transform(clean_train_phrase)

# Numpy arrays are easy to work with, so convert the result to an 

# array

train_data_features = train_data_features.toarray()

# vocab = vectorizer.get_feature_names()

# dist = np.sum(train_data_features, axis=0)

# for tag, count in zip(vocab, dist):

#     print (count, tag)
import time



start = time.time() # Start time



print("Training the Random Forest")

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=100, n_jobs = -1)

rf = rf.fit(train_data_features, train['Sentiment'])



# Get the end time and print how long the process took

end = time.time()

elapsed = end - start

print ("Time taken for K Means clustering: ", elapsed, "seconds.")
test.columns.values
num_test_phrase = test['Phrase'].size

clean_test_phrase = []

for i in range(0, num_test_phrase):

    if( (i+1)%10000 == 0 ):

        print ("Review %d of %d\n" % ( i+1, num_test_phrase ))

    clean_test_phrase.append(phrase_to_words(test['Phrase'][i]))
#Get a bag of words for the test set, and convert to a numpy array

test_data_features = vectorizer.transform(clean_test_phrase)

test_data_features = test_data_features.toarray()



#Use the random forest to make sentiment label predictions

result = rf.predict(test_data_features)



# Copy the results to a pandas dataframe with an "id" column and

# a "sentiment" column

output = pd.DataFrame( data={"PhraseId":test["PhraseId"], "Sentiment":result} )



# Use pandas to write the comma-separated output file

output.to_csv( "Sentiment_Analysis_Movie.csv", index=False, quoting=3 )