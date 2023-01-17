# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import os



import gensim



print(os.listdir("../input/quora-insincere-questions-classification/embeddings/GoogleNews-vectors-negative300/"))
import numpy as np

import pandas as pd
link = "../input/quora-insincere-questions-classification/embeddings/GoogleNews-vectors-negative300/GoogleNews-vectors-negative300.bin"



embeddings = gensim.models.KeyedVectors.load_word2vec_format(link, binary = True)
## Reading files



yelp = pd.read_csv("../input/yelp-sentiment-dataset/yelp.csv")
yelp.head()
yelp = yelp.drop("Unnamed: 0", axis=1)

yelp.columns   ## Dropping the not-utilisable column
yelp.head()
## Using Stopwords (i.e removed)##



import nltk



docs_vectors = pd.DataFrame()  ## empty dataframe

stopwords = nltk.corpus.stopwords.words('english')   ## !! added later



## in below... all lowercase shall help in covering all the words, instead of adding ""A-Z"" in RegEx which may not provide suitable outputs

for doc in yelp['review'].str.lower().str.replace('[^a-z ]', ''):

    temp = pd.DataFrame()   ## initially empty, and empty on every iteration

    for word in doc.split(' '):  ## !!

        if word not in stopwords: 

            try:

                word_vec = embeddings[word]  ## if present, the following code applies

                temp = temp.append(pd.Series(word_vec), ignore_index = True)  ## .Series to make it easier to append "without" index labels

            except:

                pass

    doc_vector = temp.mean()

    docs_vectors = docs_vectors.append(doc_vector, ignore_index = True) ## added to the empty data frame



# docs_vectors.shape ## ==> (1000 x 300) order
docs_vectors.head() ## a sparse matrix
pd.isnull(docs_vectors).sum().sum() # No null values present
## adding a column in docs_vector of "sentiment"  + dropping the null values



docs_vectors['sentiment'] = yelp['sentiment']

docs_vectors = docs_vectors.dropna()
from sklearn.model_selection import train_test_split 



## here vectorization (vectorizer) again shall not come, since we are calculated weights 

from sklearn.ensemble import AdaBoostClassifier 



train_x, test_x, train_y, test_y = train_test_split(docs_vectors.drop('sentiment', axis = 1),

                                                   docs_vectors['sentiment'],

                                                   test_size = 0.2,

                                                   random_state = 1)



train_x.shape, test_x.shape, train_y.shape, test_y.shape  ## Test and Train partitions
model = AdaBoostClassifier(n_estimators = 900, random_state = 1)

model.fit(train_x, train_y)



test_pred = model.predict(test_x)



from sklearn.metrics import accuracy_score

accuracy_score(test_y, test_pred)   



## == 77.5% accuracy score using AdaBoost algorithm (with Stopwords removed)
### Sentiment Analyzer to check out Sentiments



from nltk.sentiment import SentimentIntensityAnalyzer



sentiment = SentimentIntensityAnalyzer()
reviews = yelp['review'].str.lower().str.replace('[^a-z ]', '')

reviews   



## the Yelp reviews are put into lowercase and then using RegEx, words are split seperated.

## this format allows for better analysis of sentiment of words
yelp['sentiment'].value_counts()   
## Using a user-defined function to find out the sentiment out of Yelp reviews



def get_sentiment(text):

    sentiment = SentimentIntensityAnalyzer() #### calling Intensity Analyzer

    compound = sentiment.polarity_scores(text)['compound']  ### calling the 'compound' score for the "text" entered

    if compound > 0:

        return 1  ## positive

    else:

        return 0 ## negative

    #else:

        #return "Neutral"     

    return compound



yelp['sentiment_vader'] = yelp['review'].apply(get_sentiment) ### in the columns of "imdb"

yelp['sentiment_vader'] 
from sklearn.metrics import accuracy_score



accuracy_score(yelp['sentiment'], yelp['sentiment_vader']) ## == 80.9% of accuracy score using VADER



## ==> improved accuracy using VADER