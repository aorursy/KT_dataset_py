# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train=pd.read_csv('/kaggle/input/jigsaw-toxic-comment-classification-challenge/train.csv')
train.head()
train.tail()
label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

train['none'] = 1-train[label_cols].max(axis=1)
train.head(10)
train['comment_text'].isnull().any()
train.info()
test=pd.read_csv('/kaggle/input/jigsaw-toxic-comment-classification-challenge/test.csv')
test.isnull().any()
test.shape
stopword={'a','about','above','after','again','against','all','am','an','and','any','are','as','at','be','because','been','before',

'being','below','between','both','but','by','can',

'd',

'did',

'do',

'does',

'doing',

'down',

'during',

'each',

'few',

'for',

'from',

'further',

'had',

'has',

'hasn',

"hasn't",

'have',

'having',

'he',

'her',

'here',

'hers',

'herself',

'him',

'himself',

'his',

'how',

'i',

'if',

'in',

'into',

'is',

'it',

"it's",

'its',

'itself',

'just',

'll',

'm',

'ma',

'me',

'more',

'most',

'my',

'myself',

'no',

'now',

'o',

'of',

'off',

'on',

'once',

'only',

'or',

'other',

'our',

'ours',

'ourselves',

'out',

'over',

'own',

're',

's',

'same',

'she',

"she's",

'should',

"should've",

'so',

'some',

'such',

't',

'than',

'that',

"that'll",

'the',

'their',

'theirs',

'them',

'themselves',

'then',

'there',

'these',

'they',

'this',

'those',

'through',

'to',

'too',

'under',

'until',

'up',

've',

'very',

'was',

'we',

'were',

'what',

'when',

'where',

'which',

'while',

'who',

'whom',

'why',

'will',

'with',

'y',

'you',

"you'd",

"you'll",

"you're",

"you've",

'your',

'yours',

'yourself',

'yourselves'}
import matplotlib.pyplot as plt 

from sklearn.metrics import confusion_matrix 

from sklearn.metrics import accuracy_score

from sklearn.naive_bayes import BernoulliNB

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import train_test_split

from nltk.stem import PorterStemmer

from nltk.stem.wordnet import WordNetLemmatizer

import re 
st=PorterStemmer()
st.stem('burned')
def cleanhtml(sent):

    cleanr = re.compile('<.*?>')

    cleaned = re.sub(cleanr,' ',sent)

    return cleaned

def cleanpunc(sent):

    clean = re.sub(r'[?|!|$|#|\'|"|:]',r'',sent)

    clean = re.sub(r'[,|(|)|.|\|/]',r' ',clean)

    return clean
corpus=[]

for p in test['comment_text'].values:

    review=cleanhtml(p)

    review=cleanpunc(review)

    review=re.sub('[^a-zA-Z]',' ',review)

    review=review.lower()

    review=review.split()

    ps=PorterStemmer()

    review=[ps.stem(word) for word in review if not word in stopword]

    review=' '.join(review)

    corpus.append(review)
corpus
test.head()
test.shape
train.shape
train=train[:100000]
train.shape
corpus1=[]

for p in train['comment_text'].values:

    review=cleanhtml(p)

    review=cleanpunc(review)

    review=re.sub('[^a-zA-Z]',' ',review)

    review=review.lower()

    review=review.split()

    ps=PorterStemmer()

    review=[ps.stem(word) for word in review if not word in stopword]

    review=' '.join(review)

    corpus1.append(review)
corpus1
train.head()
train['comment_text']=corpus1
train.head()
train=train.drop('id',axis=1)
X=train['comment_text'].values
X.shape
from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.feature_extraction.text import CountVectorizer
vec = TfidfVectorizer(ngram_range=(1,2),strip_accents='unicode', use_idf=1,

               smooth_idf=1, sublinear_tf=1 )
test['comment_text']=corpus
test.head()
X_test=test['comment_text'].values
bow_train = vec.fit_transform(X)

bow_test = vec.transform(X_test)
bow_train.shape
from sklearn.naive_bayes import MultinomialNB

from sklearn.linear_model import LogisticRegression
x = bow_train

test_x = bow_test
def pr(y_i, y):

    p = x[y==y_i].sum(0)

    return (p+1) / ((y==y_i).sum()+1)
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
subm = pd.read_csv('/kaggle/input/jigsaw-toxic-comment-classification-challenge/sample_submission.csv')

submid = pd.DataFrame({'id': subm["id"]})

submission = pd.concat([submid, pd.DataFrame(preds, columns = label_cols)], axis=1)

submission.to_csv('submission.csv', index=False)
submission
