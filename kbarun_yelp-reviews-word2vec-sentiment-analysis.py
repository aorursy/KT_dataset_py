# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import os



import gensim



print(os.listdir("../input/"))

print(os.listdir("../input//embeddings/GoogleNews-vectors-negative300/"))



# Any results you write to the current directory are saved as output.
import numpy as np

import pandas as pd
link = "../input/embeddings/GoogleNews-vectors-negative300/GoogleNews-vectors-negative300.bin"



embeddings = gensim.models.KeyedVectors.load_word2vec_format(link, binary = True)
## Reading files



url = "https://raw.githubusercontent.com/skathirmani/datasets/master/yelp_labelled.csv"

yelp = pd.read_csv(url, sep = '\t')
yelp = yelp.reset_index()

yelp = yelp.rename(columns = {'index':'review','review  sentiment':'sentiment'})

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
## adding a column in docs_vector of "sentiment"  + dropping the null values



docs_vectors['sentiment'] = yelp['sentiment']

docs_vectors = docs_vectors.dropna()
# Adaptive Boost algorithm



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

print(accuracy_score(test_y, test_pred) )  



## == 77.5% accuracy score using AdaBoost algorithm (with Stopwords removed)
### Sentiment Analyzer to check out Sentiments



from nltk.sentiment import SentimentIntensityAnalyzer



sentiment = SentimentIntensityAnalyzer()
reviews = yelp['review'].str.lower().str.replace('[^a-z ]', '')

reviews.head()
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



yelp['sentiment_vader'] = yelp['review'].apply(get_sentiment) 

yelp['sentiment_vader'].head(10)
from sklearn.metrics import accuracy_score



accuracy_score(yelp['sentiment'], yelp['sentiment_vader']) 



## ==> improved accuracy using VADER