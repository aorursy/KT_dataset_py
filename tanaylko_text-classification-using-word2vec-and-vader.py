# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os



import gensim

print(os.listdir("../input/embeddings/GoogleNews-vectors-negative300/"))     

### ^^^^***** under 'embeddings' -> GoogleNews, wiki-news... these folders to use for extracting files (.bin)



# Any results you write to the current directory are saved as output.
url = "../input/embeddings/GoogleNews-vectors-negative300/GoogleNews-vectors-negative300.bin"



embeddings = gensim.models.KeyedVectors.load_word2vec_format(url, binary = True)
embeddings['sabermetrics'] ## presence of word in news
embeddings['ghuiya'] ## NOT present in news; hence, ERROR
embeddings.most_similar('camera', topn = 10)   ## based on Cosine Similarity , to find similar terms
embeddings.doesnt_match(['king', 'woman', 'gandhi', 'sonia'])  ## odd one out 



## king - man = queen
embeddings.most_similar(positive = ['king', 'woman'], negative = ['man'], topn = 10) 
url = 'https://bit.ly/2S2yXEd'  ## from the Internet, downloading a link for text-classification

imdb = pd.read_csv(url)

imdb.shape
imdb.loc[0, 'review']
embeddings['A'] 



### ^^^^^^ checking the embedding value, for which the Gradient Descent formula is used to calculate weights 
''' ## without stopwords ##

import nltk 



docs_vectors = pd.DataFrame()



## in below... all lowercase shall help in covering all the words, instead of adding ""A-Z"" in RegEx which may not provide suitable outputs

for doc in imdb['review'].str.lower().str.replace('[^a-z ]', ''):

    temp = pd.DataFrame()   ## initially empty, and empty on every iteration

    for word in nltk.word_tokenize(doc):  ## choose one word at a time from the doc from above

        try:

            word_vec = embeddings[word]  ## if present, the following code applies

            temp = temp.append(pd.Series(word_vec), ignore_index = True)  ## .Series to make it easier to append "without" index labels

        except:

            pass

    doc_vector = temp.mean()

    docs_vectors = docs_vectors.append(doc_vector, ignore_index = True) ## added to the empty data frame

docs_vectors.shape ## 300 columns is a lot lesser'''
## with stopwords ##



import nltk



docs_vectors = pd.DataFrame()

stopwords = nltk.corpus.stopwords.words('english')   ## !! added later



## in below... all lowercase shall help in covering all the words, instead of adding ""A-Z"" in RegEx which may not provide suitable outputs

for doc in imdb['review'].str.lower().str.replace('[^a-z ]', ''):

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

docs_vectors.shape ## 300 columns is a lot lesser
docs_vectors.head()
pd.isnull(docs_vectors).sum().sum() ## 600 nulls, when all in lowercase
## adding a column in docs_vector of "sentiment"  + dropping the null values

docs_vectors['sentiment'] = imdb['sentiment']

docs_vectors = docs_vectors.dropna()
from sklearn.model_selection import train_test_split ## here vectorization (vectorizer) again shall not come, since we are calculated weights 

from sklearn.ensemble import AdaBoostClassifier



train_x, test_x, train_y, test_y = train_test_split(docs_vectors.drop('sentiment', axis = 1),

                                                   docs_vectors['sentiment'],

                                                   test_size = 0.2,

                                                   random_state = 1)

train_x.shape, test_x.shape, train_y.shape, test_y.shape
model = AdaBoostClassifier(n_estimators = 800, random_state = 1)

model.fit(train_x, train_y)

test_pred = model.predict(test_x)



from sklearn.metrics import accuracy_score

accuracy_score(test_y, test_pred)  



## 74% accuracy without stopwords 



## 75.33% accuracy with stopwords
'''model = AdaBoostClassifier(n_estimators = 2000, random_state = 5)

model.fit(train_x, train_y)

test_pred = model.predict(test_x)



from sklearn.metrics import accuracy_score

accuracy_score(test_y, test_pred)  



## 74% accuracy without stopwords 



## 75.33% accuracy with stopwords''' ## accuracy fell to 74.6% with stopwords
from nltk.sentiment import SentimentIntensityAnalyzer



sentiment = SentimentIntensityAnalyzer()
reviews = imdb['review'].str.lower().str.replace('[^a-z ]', '')

reviews
imdb['sentiment'].value_counts()
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



imdb['sentiment_vader'] = imdb['review'].apply(get_sentiment) ### in the columns of "imdb"

imdb['sentiment_vader'] 
get_sentiment("YES")
from sklearn.metrics import accuracy_score



accuracy_score(imdb['sentiment'], imdb['sentiment_vader']) ## == 79.011% of accuracy score