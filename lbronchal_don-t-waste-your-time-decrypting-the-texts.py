import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from collections import Counter
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import os
print(os.listdir("../input"))
data = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
subm = pd.read_csv('../input/sample_submission.csv')
data.head()
data_1 = data.query('difficulty==1')
data_1.head()
alp = pd.Series(Counter(''.join(data_1['ciphertext'])))
alp.head(10)
len(alp)
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import FunctionTransformer
from sklearn.model_selection import GridSearchCV
X = data_1.drop('target', axis=1)
y = data_1['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, stratify=y, random_state=0)
def tokenize(text): 
    return text.split("1")

vectorizer = CountVectorizer(
    analyzer = 'word',
    tokenizer = tokenize,
    lowercase = False,
    ngram_range=(1, 1))

estimator = LogisticRegression(random_state=0)
model = Pipeline([('selector', FunctionTransformer(lambda x: x['ciphertext'], validate=False)),
                  ('vectorizer', vectorizer), 
                  ('tfidf', TfidfTransformer()),
                  ('estimator', estimator)])
def generate_tokenizer(separator):
    def tokenizer(text):
        return text.split(separator)
    return tokenizer
tokenize_1 = generate_tokenizer("1")

model.steps[1][1].set_params(tokenizer=tokenize_1)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
f1_score(y_test, y_pred, average='macro')
def evaluate_delimiters(data):    
    X = data.drop('target', axis=1)
    y = data['target']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=.2, stratify=y, random_state=0)
    
    scores = {}
    
    # let's get all the chars that are used:
    alp = pd.Series(Counter(''.join(data['ciphertext'])))

    for c in alp.keys():
        tokenize = generate_tokenizer(c)
        model.steps[1][1].set_params(tokenizer=tokenize)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        score = f1_score(y_test, y_pred, average='macro')
        scores[c] = score
    return pd.Series(scores).sort_values(ascending=False)
scores_difficulty_1 = evaluate_delimiters(data.query('difficulty==1'))
scores_difficulty_1[:10]
scores_difficulty_2 = evaluate_delimiters(data.query('difficulty==2'))
scores_difficulty_2[:10]
scores_difficulty_3 = evaluate_delimiters(data.query('difficulty==3'))
scores_difficulty_3[:10]
scores_difficulty_4 = evaluate_delimiters(data.query('difficulty==4'))
scores_difficulty_4[:10]
book = {'1': ' ',
 '\x1b': 'e',
 't': 't',
 'O': 'a',
 '^': 'o',
 'a': 'i',
 '\x02': 'n',
 'v': 's',
 '#': 'r',
 '0': 'h',
 '8': 'l',
 's': '\n',
 'A': 'd',
 '_': 'c',
 'c': 'u',
 '-': 'm',
 '\x08': '.',
 'q': '-',
 "'": 'p',
 'd': 'g',
 'o': 'y',
 ']': 'f',
 'W': 'w',
 '\x03': 'b',
 'T': ',',
 'z': 'v',
 ':': 'I',
 '[': '>',
 'f': 'k',
 'G': ':',
 'L': '1',
 '>': 'S',
 '{': 'T',
 '/': 'A',
 '\\': '0',
 '2': 'C',
 'y': ')',
 'e': 'M',
 ';': "'",
 '|': '(',
 'Z': '=',
 'H': '2',
 '\x1c': '*',
 '\x1e': 'R',
 'x': 'D',
 '\x7f': 'N',
 '%': 'O',
 'Q': '\t',
 '9': 'P',
 'E': 'E',
 'F': 'L',
 ')': 'E',
 'u': '3',
 'b': '@',
 'J': 'B',
 '6': '"',
 'g': 'H',
 '*': 'F',
 '<': '9',
 '\t': '5',
 ',': '4',
 '+': 'x',
 'l': 'W',
 'X': 'j',
 '5': '6',
 '"': 'G',
 'n': '8',
 '@': 'U',
 '&': '?',
 'h': 'z',
 '?': '/',
 '\x06': '7',
 '}': 'J',
 '4': 'J',
 'P': '!',
 'w': 'K',
 '\x18': 'V',
 '\x10': 'Y',
 '!': 'X',
 '(': 'Y',
 ' ': '<',
 '\x1a': 'q',
 '`': '>',
 '.': '#',
 'B': '$',
 '~': '+',
 '3': ';',
 'V': 'Q',
 'm': 'q',
 '\x0c': '%',
 'U': '[',
 'i': ']',
 'r': '&',
 'K': 'Z',
 'Y': '~',
 'I': '}',
 'k': '{',
 'S': '\r',
 '$': '\x08',
 'p': '\x02'}

dec_table = str.maketrans(book)
data_1_clean = data.query('difficulty==1').copy()
data_1_clean['ciphertext'] = data_1_clean['ciphertext'].map(lambda x: x.translate(dec_table))
data_1_clean['ciphertext'][1]
X = data_1_clean.drop('target', axis=1)
y = data_1_clean['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, stratify=y, random_state=0)
tokenize_ws = generate_tokenizer(" ")
model.steps[1][1].set_params(tokenizer=tokenize_ws)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
f1_score(y_test, y_pred, average='macro')