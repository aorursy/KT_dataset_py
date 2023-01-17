# import packages
import numpy as np
import pandas as pd
import re

from gensim import corpora, models, similarities

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

np.random.seed(27)

import warnings
warnings.filterwarnings("ignore")
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
train.head()
contractions = {
"ain't": "is not",
"aren't": "are not",
"can't": "cannot",
"can't've": "cannot have",
"'cause": "because",
"could've": "could have",
"couldn't": "could not",
"couldn't've": "could not have",
"didn't": "did not",
"doesn't": "does not",
"don't": "do not",
"hadn't": "had not",
"hadn't've": "had not have",
"hasn't": "has not",
"haven't": "have not",
"he'd": "he would",
"he'd've": "he would have",
"he'll": "he will",
"he'll've": "he he will have",
"he's": "he is",
"how'd": "how did",
"how'd'y": "how do you",
"how'll": "how will",
"how's": "how is",
"I'd": "I would",
"I'd've": "I would have",
"I'll": "I will",
"I'll've": "I will have",
"I'm": "I am",
"I've": "I have",
"i'd": "i would",
"i'd've": "i would have",
"i'll": "i will",
"i'll've": "i will have",
"i'm": "i am",
"i've": "i have",
"isn't": "is not",
"it'd": "it would",
"it'd've": "it would have",
"it'll": "it will",
"it'll've": "it will have",
"it's": "it is",
"let's": "let us",
"ma'am": "madam",
"mayn't": "may not",
"might've": "might have",
"mightn't": "might not",
"mightn't've": "might not have",
"must've": "must have",
"mustn't": "must not",
"mustn't've": "must not have",
"needn't": "need not",
"needn't've": "need not have",
"o'clock": "of the clock",
"oughtn't": "ought not",
"oughtn't've": "ought not have",
"shan't": "shall not",
"sha'n't": "shall not",
"shan't've": "shall not have",
"she'd": "she would",
"she'd've": "she would have",
"she'll": "she will",
"she'll've": "she will have",
"she's": "she is",
"should've": "should have",
"shouldn't": "should not",
"shouldn't've": "should not have",
"so've": "so have",
"so's": "so as",
"that'd": "that would",
"that'd've": "that would have",
"that's": "that is",
"there'd": "there would",
"there'd've": "there would have",
"there's": "there is",
"they'd": "they would",
"they'd've": "they would have",
"they'll": "they will",
"they'll've": "they will have",
"they're": "they are",
"they've": "they have",
"to've": "to have",
"wasn't": "was not",
"we'd": "we would",
"we'd've": "we would have",
"we'll": "we will",
"we'll've": "we will have",
"we're": "we are",
"we've": "we have",
"weren't": "were not",
"what'll": "what will",
"what'll've": "what will have",
"what're": "what are",
"what's": "what is",
"what've": "what have",
"when's": "when is",
"when've": "when have",
"where'd": "where did",
"where's": "where is",
"where've": "where have",
"who'll": "who will",
"who'll've": "who will have",
"who's": "who is",
"who've": "who have",
"why's": "why is",
"why've": "why have",
"will've": "will have",
"won't": "will not",
"won't've": "will not have",
"would've": "would have",
"wouldn't": "would not",
"wouldn't've": "would not have",
"y'all": "you all",
"y'all'd": "you all would",
"y'all'd've": "you all would have",
"y'all're": "you all are",
"y'all've": "you all have",
"you'd": "you would",
"you'd've": "you would have",
"you'll": "you will",
"you'll've": "you will have",
"you're": "you are",
"you've": "you have"
}

c_re = re.compile('(%s)' % '|'.join(contractions.keys()))

def expandContractions(text, c_re=c_re):
    def replace(match):
        return contractions[match.group(0)]
    return c_re.sub(replace, text)
# function to clean and lemmatize text and remove stopwords
from gensim.parsing.preprocessing import preprocess_string

def gensim_preprocess(docs):
    docs = [expandContractions(doc) for doc in docs]
    docs = [preprocess_string(text) for text in docs]
    return pd.Series(docs)

gensim_preprocess(train.question_text.iloc[10:15])
# apply text-preprocessing function to training set
%time train_corpus = gensim_preprocess(train.question_text)
def create_ngrams(docs):
    # create the bigram and trigram models
    bigram = models.Phrases(docs, min_count=1, threshold=1)
    trigram = models.Phrases(bigram[docs], min_count=1, threshold=1)  
    # phraser is faster
    bigram_mod = models.phrases.Phraser(bigram)
    trigram_mod = models.phrases.Phraser(trigram)
    # apply to docs
    docs = trigram_mod[bigram_mod[docs]]
    return docs

train_texts = create_ngrams(train_corpus)
train_texts[81]
# preparing ngrams for modeling
train_texts = [' '.join(text) for text in train_texts]
train['ngrams'] = train_texts
from sklearn.feature_extraction.text import TfidfVectorizer

tv = TfidfVectorizer(use_idf=True,
                     min_df=50,
                     max_features=20000,
                     ngram_range=(1,2),
                     norm='l1',
                     smooth_idf=True).fit(train_texts)
tv_matrix = tv.transform(train_texts)
# print target counts
train.target.value_counts()
# Upsampling
from imblearn.over_sampling import RandomOverSampler
ros = RandomOverSampler(sampling_strategy={1:1225312, # upsample minority class to equal majority count
                                          })
X, y = ros.fit_sample(tv_matrix, train.target)

# split into train/test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X_train, y_train)

y_ = lr.predict(X_test)
print('Logistic Regression Score: ', f1_score(y_, y_test))

print(classification_report(y_test, y_))
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, y_)
from sklearn.naive_bayes import BernoulliNB

bnb = BernoulliNB()
%time bnb.fit(X_train, y_train)
y_ = bnb.predict(X_test)

print('Naive Bayes Score: ', f1_score(y_, y_test))

print(classification_report(y_test, bnb_y_))

pd.DataFrame(confusion_matrix(y_test, bnb_y_))
import xgboost as xgb

xgb_model = xgb.XGBClassifier().fit(X_train, y_train)
y_ = xgb_model.predict(X_test)

print('XGBoost Score: ', f1_score(y_, y_test))

print(classification_report(y_test, xgb_y_))

pd.DataFrame(confusion_matrix(y_test, xgb_y_))
from sklearn.ensemble import VotingClassifier

#create submodels
estimators = []

model1 = lr
model2 = bnb
model3 = xgb_model


estimators.append(('logistic', model1))
estimators.append(('bernoulli', model2))
estimators.append(('xgboost', model3))
# create ensemble model
%time ensemble = VotingClassifier(estimators).fit(X_train, y_train)
y_ = ensemble.predict(X_test)
print('Ensemble Score: ', f1_score(y_, y_test))
%time ensemble_y_ = ensemble.predict(X_test)
print(classification_report(y_test, ensemble_y_))

pd.DataFrame(confusion_matrix(y_test, ensemble_y_))
# preprocessing/lemmatizing/stemming test data
%time test_corpus = gensim_preprocess(test.question_text)
test_texts = create_ngrams(test_corpus)

test_texts = [' '.join(text) for text in test_texts]
test['ngrams'] = test_texts
test.head()
# ensemble on test data
# fit on whole training set
ensemble.fit(tv_matrix, train.target)
prediction = ensemble.predict(tv.transform(test.ngrams))

submission = pd.DataFrame({'qid':test.qid, 'prediction':prediction})
submission.to_csv('submission.csv', index=False)
submission.head()