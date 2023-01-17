# -------------------------------------------------------START-----------------------------------------

"""It's my first nlp pratice my code is mainly based on use datacamp nlp module, 
kirill eremenko machine learning mooc section on nlp
and the great notebook of Bojan Tuguz "logistic regression with word and char n-grams.
thanks to them, i'm open to criticism in purpose to improve myself"""

import numpy as np
import pandas as pd 
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score
from scipy.sparse import hstack
from sklearn.feature_extraction.text import HashingVectorizer
import re 
import nltk
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import nltk
#nltk.download('stopwords')
  

# --------------------------------------------------------------------------------------------

train = pd.read_csv('../input/train.csv')[:50000]
test = pd.read_csv('../input/test.csv')[:50000]

#train.head()


train['comment_text'] = train['comment_text'].fillna('__nocomment__')
test['comment_text']  = test['comment_text'].fillna('__nocomment__')

#seems those line are too greedy for kaggle so i use 10k rows

train_text = train["comment_text"]
test_text = test["comment_text"]

#test.shape
#train.shape




#--------------------------------------------------------------------------


ps = PorterStemmer()

#word 

train_corpus = []
for i in range(0, 50000):
    data = re.sub("[^a-zA-Z]", ' ', train_text[i]).lower().split()
    data = [ps.stem(word) for word in data if not word in set(stopwords.words("english"))]
    data = ' '.join(data)
    train_corpus.append(data)



test_corpus = []

for i in range(0, 50000):
    data2 = re.sub("[^a-zA-Z]", ' ', test_text[i]).lower().split()
    data2 = [ps.stem(word) for word in data2 if not word in set(stopwords.words("english"))]
    data2 = ' '.join(data2)
    test_corpus.append(data2)


#-------------------------------------------------- char/word --------------------

word_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='word',
    ngram_range=(1, 2),
    max_features=8000,
    norm = "l1") 
word_vectorizer.fit(train_corpus)

word_vectorizer2 = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='word',
    ngram_range=(1, 2),
    max_features=8000,
    norm = "l1")

word_vectorizer2.fit(test_corpus)

train_word_features = word_vectorizer.transform(train_corpus)
test_word_features = word_vectorizer2.transform(test_corpus)

#~~~~~~~~

char_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='char',
    ngram_range=(4, 6),
    max_features=25000, 
    norm = "l1")
char_vectorizer.fit(train_corpus)

char_vectorizer2 = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='char',
    stop_words='english',
    ngram_range=(4, 6),
    max_features=25000, 
    norm="l1")

char_vectorizer2.fit(test_corpus)

train_char_features = char_vectorizer.transform(train_corpus)
test_char_features = char_vectorizer2.transform(test_corpus)

#----------------------------------------------------

train_features = hstack([train_char_features, train_word_features])
test_features = hstack([test_char_features, test_word_features])
#-----------------------------------------------

#submission template took from Dr.fuzzy notebook

class_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

Scores = []

submission = pd.DataFrame.from_dict({'id': test['id']})

for class_name in class_names:
    train_target = train[class_name]
    classifier = MultinomialNB()
    cv_score = np.mean(cross_val_score(classifier, train_features, train_target, cv=5, scoring='roc_auc'))
    Scores.append(cv_score)
    print('CV score for class {} is {}'.format(class_name, cv_score))
    classifier.fit(train_features, train_target)
    submission[class_name] = classifier.predict_proba(test_features)[:, 1]
    
#submission.to_csv('submission.csv', index=False)

# ------------------------------------------------- End -------------------------------------------