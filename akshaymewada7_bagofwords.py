# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/labeledTrainData.tsv',sep='\t')

data.head()
data['review'].tail()
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

for r in data['review']:

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
print(tfidf,data['sentiment'])
from sklearn.cross_validation import train_test_split

train_x,test_x,train_y,test_y = train_test_split(tfidf,data['sentiment'])
from sklearn.svm import SVC

clf = SVC(C=10,gamma=0.01)

clf = clf.fit(train_x, train_y)
train_y.head()
print(clf.score(test_x, test_y))