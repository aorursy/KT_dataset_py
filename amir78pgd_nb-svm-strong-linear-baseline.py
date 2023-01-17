import sys

sys.setrecursionlimit(5000)
import os

os.listdir('../input/jigsaw-toxic-comment-classification-challenge/')
import pandas as pd, numpy as np

from sklearn.linear_model import LogisticRegression

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
train = pd.read_csv('../input/jigsaw-toxic-comment-classification-challenge/train.csv')

test = pd.read_csv('../input/jigsaw-toxic-comment-classification-challenge/test.csv')

subm = pd.read_csv('../input/jigsaw-toxic-comment-classification-challenge/sample_submission.csv')
train.head()
train['comment_text'][0]
train['comment_text'][2]
lens = train.comment_text.str.len()

lens.mean(), lens.std(), lens.max()
lens.hist();
label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

train['none'] = 1-train[label_cols].max(axis=1)

train.describe()
len(train),len(test)
COMMENT = 'comment_text'

train[COMMENT].fillna("unknown", inplace=True)

test[COMMENT].fillna("unknown", inplace=True)
import nltk

from nltk.corpus import stopwords

from nltk.tokenize import word_tokenize

import re
REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')

BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')

STOPWORDS = set(stopwords.words('english'))



def text_prepare(text): ### The function will take in text and lower case it remove the stopwords, symbols and return it

    

    #import pdb; pdb.set_trace()

    text = text.lower()                          ### Write a code which can change the input text to lowercase

    text = REPLACE_BY_SPACE_RE.sub(' ', text)    ### Write a code which replaces REPLACE_BY_SPACE_RE symbols by space in text

    text = BAD_SYMBOLS_RE.sub('', text)          ### Write a code which deletes symbols which are in BAD_SYMBOLS_RE from text

    pattern = re.compile(r'\b(' + r'|'.join(STOPWORDS) + r')\b\s*') 

    text = pattern.sub('', text)                 ### Write a code which deletes stopwords from text

    text = re.sub(' +', ' ', text)

        

    return text
train[COMMENT]
test[COMMENT]
train[COMMENT] = [text_prepare(COMMENT) for COMMENT in train[COMMENT]]

test[COMMENT] = [text_prepare(COMMENT) for COMMENT in test[COMMENT]]
import nltk

from nltk.stem import PorterStemmer, WordNetLemmatizer
#create objects for stemmer and lemmatizer

lemmatiser = WordNetLemmatizer()

stemmer = PorterStemmer()

#download words from wordnet library

#nltk.download('wordnet')
import string

print(string.punctuation)

punctuation_edit = string.punctuation.replace('\'','') +"0123456789"

print (punctuation_edit)

outtab = "                                         "

trantab = str.maketrans(punctuation_edit, outtab)
os.listdir('../input/wordnet/wordnet/')
train_comment = train[COMMENT]

train_comment_mat = train_comment.as_matrix()
test_comment = test[COMMENT]

test_comment_mat = test_comment.as_matrix()
%%time



for i in range(len(train_comment)):

    train_comment[i] = train_comment[i].lower().translate(trantab)

    l = []

    for word in train_comment[i].split():

        l.append(stemmer.stem(lemmatiser.lemmatize(word,pos="v")))

    train_comment[i] = " ".join(l)
for i in range(len(test_comment)):

    test_comment[i] = test_comment[i].lower().translate(trantab)

    l = []

    for word in test_comment[i].split():

        l.append(stemmer.stem(lemmatiser.lemmatize(word,pos="v")))

    test_comment[i] = " ".join(l)
import re, string

re_tok = re.compile(f'([{string.punctuation}“”¨«»®´·º½¾¿¡§£₤‘’])')

def tokenize(s): return re_tok.sub(r' \1 ', s).split()
n = train.shape[0]

vec = TfidfVectorizer(ngram_range=(1,2), tokenizer=tokenize,

               min_df=3, max_df=0.9, strip_accents='unicode', use_idf=1,

               smooth_idf=1, sublinear_tf=1 )

#trn_term_doc = vec.fit_transform(train[COMMENT])

#test_term_doc = vec.transform(test[COMMENT])

trn_term_doc = vec.fit_transform(train_comment)

test_term_doc = vec.transform(test_comment)
trn_term_doc, test_term_doc
def pr(y_i, y):

    p = x[y==y_i].sum(0)

    return (p+1) / ((y==y_i).sum()+1)
x = trn_term_doc

test_x = test_term_doc
def get_mdl(y):

    y = y.values

    r = np.log(pr(1,y) / pr(0,y))

    m = LogisticRegression(C=4, dual=True)

    x_nb = x.multiply(r)

    return m.fit(x_nb, y), r
preds = np.zeros((len(test), len(label_cols)))



for i, j in enumerate(label_cols):

    print('fit', j)

    m,r = get_mdl(train[j])

    preds[:,i] = m.predict_proba(test_x.multiply(r))[:,1]
submid = pd.DataFrame({'id': subm["id"]})

submission = pd.concat([submid, pd.DataFrame(preds, columns = label_cols)], axis=1)

submission.to_csv('submission.csv', index=False)