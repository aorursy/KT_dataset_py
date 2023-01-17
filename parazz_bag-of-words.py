from sklearn.utils import shuffle

import pandas as pd  



train = pd.read_csv("../input/labeledTrainData.tsv",  delimiter="\t", quoting=3, header=0)



train = shuffle(train)
train.shape
train
from bs4 import BeautifulSoup 

import lxml

import re

from nltk.corpus import stopwords 
def review_to_words( raw_review ):

    review_text = BeautifulSoup(raw_review, "html.parser").get_text()        

    letters_only = re.sub("[^a-zA-Z]", " ", review_text) 

    words = letters_only.lower().split()                             

    stops = set(stopwords.words("english"))                  

    meaningful_words = [w for w in words if not w in stops]   

    return( " ".join( meaningful_words )) 
clean_train_reviews = []

for r in train['review']:

    clean_train_reviews.append(review_to_words(r))
len(clean_train_reviews)
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfTransformer



vectorizer = CountVectorizer(analyzer = "word",   \

                             tokenizer = None,    \

                             preprocessor = None, \

                             stop_words = None,   \

                             max_features = 5000) 



train_data_features = vectorizer.fit_transform(clean_train_reviews)



transformer = TfidfTransformer(smooth_idf=False)



tfidf = transformer.fit_transform(train_data_features)

train_x = tfidf[:2000]

train_y = train['sentiment'][:2000]



test_x = tfidf[2000:]

test_y = train['sentiment'][2000:]
from sklearn.ensemble import RandomForestClassifier

from sklearn.neural_network import BernoulliRBM



clf = RandomForestClassifier(n_estimators = 200) 

clf = clf.fit(train_x, train_y)
train_y
print("Mean Accuracy:")

print(clf.score(test_x, test_y))