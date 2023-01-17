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
dataTrain = pd.read_csv("../input/train.csv")

dataTrain.head(5)
import seaborn as sns

sns.countplot(dataTrain['author'])
from sklearn.model_selection import train_test_split

train, test = train_test_split(dataTrain, test_size = 0.3)
from bs4 import BeautifulSoup

import nltk

from nltk.corpus import stopwords

from nltk.stem.porter import PorterStemmer

english_stemmer=nltk.stem.SnowballStemmer('english')

import re

def text_to_wordlist( text, remove_stopwords=True):

    # Function to convert a document to a sequence of words,

    # optionally removing stop words.  Returns a list of words.

    #

    # 1. Remove HTML

    review_text = BeautifulSoup(review, "lxml").get_text()



    #

    # 2. Remove non-letters

    review_text = re.sub("[^a-zA-Z]"," ", review)

    #

    # 3. Convert words to lower case and split them

    words = review_text.lower().split()

    #

    # 4. Optionally remove stop words (True by default)

    if remove_stopwords:

        stops = set(stopwords.words("english"))

        words = [w for w in words if not w in stops]



    b=[]

    stemmer = english_stemmer #PorterStemmer()

    for word in words:

        b.append(stemmer.stem(word))



    # 5. Return a list of words

    return(b)
clean_train_reviews = []

for review in train['text']:

    clean_train_reviews.append( " ".join(text_to_wordlist(review)))

    

clean_test_reviews = []

for review in test['text']:

    clean_test_reviews.append( " ".join(text_to_wordlist(review)))
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer( min_df=2, max_df=0.95, max_features = 200000, ngram_range = ( 1, 4 ),

                              sublinear_tf = True )



vectorizer = vectorizer.fit(clean_train_reviews)



train_features = vectorizer.transform(clean_train_reviews)

test_features = vectorizer.transform(clean_test_reviews)
from sklearn.feature_selection.univariate_selection import SelectKBest, chi2, f_classif

fselect = SelectKBest(chi2 , k=10000)

train_features = fselect.fit_transform(train_features, train["author"])

test_features = fselect.transform(test_features)
from sklearn.linear_model import SGDClassifier, SGDRegressor

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.naive_bayes import MultinomialNB



model1 = MultinomialNB(alpha=0.001)

model1.fit( train_features, train["author"] )



model2 = SGDClassifier(loss='modified_huber', n_iter=5, random_state=0, shuffle=True)

model2.fit( train_features, train["author"] )



model3 = RandomForestClassifier()

model3.fit( train_features, train["author"] )



model4 = GradientBoostingClassifier()

model4.fit( train_features, train["author"] )



pred_1 = model1.predict( test_features.toarray() )

pred_2 = model2.predict( test_features.toarray() )

pred_3 = model3.predict( test_features.toarray() )

pred_4 = model4.predict( test_features.toarray() )
from sklearn.metrics import accuracy_score

print('prediction 1 accuracy: ', accuracy_score(test['author'], pred_1))

print('prediction 2 accuracy: ', accuracy_score(test['author'], pred_2))

print('prediction 3 accuracy: ', accuracy_score(test['author'], pred_3))

print('prediction 4 accuracy: ', accuracy_score(test['author'], pred_4))