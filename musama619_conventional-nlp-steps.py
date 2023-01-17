import numpy as np

import pandas as pd

import re
train = pd.read_csv('../input/nlp-getting-started/train.csv')
train
train = train.drop(['keyword', 'location'], axis = 1)
train.isnull().sum()
import re #remove punctuations

import nltk

nltk.download('all')

nltk.download('stopwords')

from nltk.corpus import stopwords #and, in, the, a ... etc

from nltk.stem import WordNetLemmatizer
corpus = []

for i in range(0, 7613):

    review_train = re.sub(r"[^a-zA-Z]", ' ', train['text'][i]) #replace anything that is not a letter like "" , ... with space

    review_train = re.sub(r'http\S+', r'', review_train)

    review_train = re.sub(r'#([^\s]+)', r'\1', review_train)  

    review_train = re.sub('[\s]+', ' ', review_train)



    review_train = review_train.lower()  #all letters to lower-case

    review_train = review_train.split() #splitting review in diferent words

    



    lemmatizer = WordNetLemmatizer()

    stopwords1 = stopwords.words('english')

    

    review_train = [lemmatizer.lemmatize(word) for word in review_train if not word in stopwords1]



    review_train = ' '.join(review_train) #separating words with space and joining

    corpus.append(review_train)
#corpus
test = pd.read_csv('../input/nlp-getting-started/test.csv')
corpus_test = []

for i in range(0, 3263):

    review = re.sub(r"[^a-zA-Z]", ' ', test['text'][i]) #replace anything that is not a letter like "" , ... with space

    review = re.sub(r'http\S+', r'', review)

    review = re.sub(r'#([^\s]+)', r'\1', review)

    review = re.sub('[\s]+', ' ', review)



    review = review.lower()  #all letters to lower-case

    review = review.split() #splitting review in diferent words

    

    lemmatizer_2 = WordNetLemmatizer()

    stopwords_2 = stopwords.words('english')

    

    review = [lemmatizer_2.lemmatize(word) for word in review if not word in stopwords_2]



    review = ' '.join(review) #separating words with space and joining

    corpus_test.append(review)
#corpus_test
from sklearn.feature_extraction.text import TfidfVectorizer
vect = TfidfVectorizer(min_df=2 ,max_features = None,analyzer="word",  ngram_range=(1,3))
X_vect = vect.fit_transform(corpus)
y_vect = train['target']
X_vect.shape
y_vect.shape
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X_vect, y_vect, test_size=0.2, random_state=1)
from sklearn.linear_model import LogisticRegression

model_1 = LogisticRegression(penalty = 'l2', C=3, max_iter = 550)
model_1.fit(x_train, y_train)
model_1.score(x_test, y_test)
X_vect_test = vect.transform(corpus_test)
predict_n = model_1.predict(X_vect_test)
test_id = test['id']
submission_2 = pd.DataFrame({'id':test_id, 'target':predict_n})
submission_2
submission_2.to_csv('./NLP_start_9.csv', index=False)