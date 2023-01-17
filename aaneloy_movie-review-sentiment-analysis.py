# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import svm
from scipy import sparse

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score,make_scorer
from sklearn.model_selection import StratifiedShuffleSplit

from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer

from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import ADASYN
train = pd.read_csv("../input/movie-review-sentiment-analysis-kernels-only/train.tsv", delimiter = '\t')
test = pd.read_csv("../input/movie-review-sentiment-analysis-kernels-only/test.tsv", delimiter = '\t')
submission = pd.read_csv("../input/movie-review-sentiment-analysis-kernels-only/sampleSubmission.csv",sep='delimiter', header=None)
train.head(10)
test.head()
y_train = train['Sentiment']
sns.countplot(y_train)
class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()
    def __call__(self, doc):
        return [WordNetLemmatizer().lemmatize(w) for w in word_tokenize(doc)]
import nltk
nltk.download('punkt')
nltk.download('wordnet')

from nltk import word_tokenize,sent_tokenize
vectorizer_w = TfidfVectorizer(sublinear_tf = True, max_df = 0.5, stop_words = None,ngram_range = (1,3), analyzer = 'word', encoding = 'utf-8', tokenizer = LemmaTokenizer())
vectorizer_c = TfidfVectorizer(sublinear_tf = True, max_df = 0.5, stop_words = None,ngram_range = (2,6), analyzer = 'char', encoding = 'utf-8', tokenizer = LemmaTokenizer())
X_train_w = vectorizer_w.fit_transform(train['Phrase'])
X_train_c = vectorizer_c.fit_transform(train['Phrase'])
X_test_w = vectorizer_w.transform(test['Phrase'])
X_test_c = vectorizer_c.transform(test['Phrase'])
X_train = sparse.hstack([X_train_w, X_train_c])
X_test = sparse.hstack([X_test_w, X_test_c])

#Tried Oversampling methods using imbalanced-learn API(http://contrib.scikit-learn.org/imbalanced-learn/stable/api.html)
#However Oversampling did not help
ros = RandomOverSampler(random_state=42)
ada = ADASYN(random_state=152)
#X_train_ros, y_train_ros = ros.fit_sample(X_train, y_train)
#X_train_ada, y_train_ada = ada.fit_sample(X_train, y_train)
print("Number of samples in Train dataset i.e. n_samples: %d, Number of features in Train dataset i.e. n_features: %d" % X_train.shape)
print("Number of samples in Test dataset i.e. n_samples: %d, Number of features in Test dataset i.e. n_features: %d" % X_test.shape)
print("\n")
#print("Number of samples in Resample Train dataset(Ramdom Sampler) i.e. n_samples: %d, Number of features in Train dataset i.e. n_features: %d" % X_train_ros.shape)
#print("Number of samples in Resample Train dataset(ADASYN) i.e. n_samples: %d, Number of features in Train dataset i.e. n_features: %d" % X_train_ada.shape)
clf = MultinomialNB()
clf.fit(X_train,y_train)
y_pred_nb = clf.predict(X_test)
submission.to_csv("sampleSubmission.csv", index = False)
lclf = LogisticRegression(solver = 'saga',multi_class = 'multinomial', max_iter = 500, 
                          C = 4, random_state = 42, verbose = 10, class_weight = 'balanced')

#parameters = {'C':[2 , 4] }
#scorer = make_scorer(accuracy_score)
#cv = StratifiedShuffleSplit(2, random_state = 62)
#grid_obj = GridSearchCV(lclf, param_grid=parameters, cv = cv, scoring=scorer, n_jobs=-1, verbose=10)
#grid_fit = grid_obj.fit(X_train, y_train)
#best_clf = grid_fit.best_estimator_

predictions = (lclf.fit(X_train, y_train)).predict(X_test)
#best_predictions = best_clf.predict(X_tes
#Standard NLP Pre-Processing
X_train = train['Phrase']
X_test = test['Phrase']
X_train.head()
#NORMALIZATION - Converting to lower case
X_train_l = X_train.str.lower()
print(X_train_l[0])
X_train_l.head()
#NORMALIZATION - Removing Punctuation marks
import re
def punc_rem(y):
    return re.sub(r"[^a-zA-Z0-9]", " ", y)
X_train_p = X_train_l.apply(lambda x: punc_rem(x))
print(X_train_p[0])
X_train_p.head()
#TOKENIZATION - Word & Setence tokenizers
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize

X_train_wt = X_train_p.apply(lambda x : word_tokenize(x))
X_train_st = X_train_p.apply(lambda x : sent_tokenize(x))

print(X_train_wt[0])
print(X_train_wt.head())
print(X_train_st[0])
print(X_train_st.head())
#STOPWORDS Removal

nltk.download('stopwords')
from nltk.corpus import stopwords
print(stopwords.words('english'))
def stop_words(x):
    return [i for i in x if i not in stopwords.words('english')]
X_train_sw = X_train_wt.apply(lambda x : stop_words(x))
print(X_train_sw[0])
print(X_train_sw[3])
print(X_train_sw.head())
#POS (Parts Of Speech Tagging) & NER (Named Entity Recognition)

import nltk
nltk.download('averaged_perceptron_tagger')
def postag(x):
    return nltk.pos_tag(x)
X_train_pos = X_train_wt.apply(lambda x: postag(x))
print(X_train_pos[0])
print(X_train_pos[3])
#nltk.help.upenn_tagset('CC')
nltk.download('tagsets')
for i in X_train_pos[0]:
    print("{}: ".format(i))
    (nltk.help.upenn_tagset(i[1]))
nltk.corpus.stopwords.readme() #https://www.nltk.org/book/ch05.html
nltk.download('maxent_ne_chunker')
nltk.download('words')
def ner(x):
    return nltk.ne_chunk(x)

X_train_ner = X_train_pos.apply(lambda x: ner(x))

print(X_train_ner[0])
print(nltk.ne_chunk(nltk.pos_tag(word_tokenize("Bangladesh is a great country"))))
#https://www.nltk.org/book/ch07.html
#CFG - Context Free Grammer

print(X_train[1])
print(X_train_wt[1])
print(X_train_pos[1])
nltk.help.upenn_tagset('JJ')
custom_grammer = nltk.CFG.fromstring("""
S -> NP VP
PP -> P NP
NP -> Det N | Det N PP 
VP -> V NP | VP PP | JJ
Det -> 'the'|'a'
N -> 'series'|'escapades'|'adage'|'goose'
V -> 'demonstrating'|'is'
JJ -> 'good'
P -> 'that'|'for'|'of'|'what'
""")

custom_parser = nltk.ChartParser(custom_grammer)
print(custom_parser.parse(X_train_wt[1]))
help(nltk.ChartParser)
for custom_tree in custom_parser.parse(X_train_wt[1]):
    print("Bangladesh")
# Define a custom grammar
my_grammar = nltk.CFG.fromstring("""
S -> NP VP
PP -> P NP
NP -> Det N | Det N PP | 'I'
VP -> V NP | VP PP
Det -> 'an' | 'my'
N -> 'elephant' | 'pajamas'
V -> 'shot'
P -> 'in'
""")
parser = nltk.ChartParser(my_grammar)
nltk.pos_tag(sentence)
# Parse a sentence
sentence = word_tokenize("I shot an elephant in my pajamas")
print(type(sentence))
nltk.pos_tag(sentence)
print(parser.parse(sentence))
for tree in parser.parse(sentence):
    print(type(tree))
    print(tree)
#STEMMING and LEMMATIZATION
from nltk.stem import porter
stemmer = porter.PorterStemmer()
def stmr(x):
    return [stemmer.stem(i) for i in x]
X_train_stm = X_train_wt.apply(lambda x: stmr(x))
print(X_train_wt[0])
print(X_train_stm[0])
from nltk.stem.wordnet import WordNetLemmatizer
def lmtr(x):
    return [WordNetLemmatizer().lemmatize(i) for i in x]

def lmtrv(x):
    return [WordNetLemmatizer().lemmatize(i, pos = 'v') for i in x]

X_train_lm = X_train_wt.apply(lambda x: lmtr(x))
X_train_lmv = X_train_wt.apply(lambda x: lmtrv(x))
print(X_train_wt[1])
print(X_train_lm[1])
print(X_train_lmv[1])