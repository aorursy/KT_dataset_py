import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_union
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import learning_curve
from sklearn.feature_selection import SelectFromModel

from itertools import product
from sklearn.ensemble import VotingClassifier
from sklearn.naive_bayes import ComplementNB
from scipy.sparse import csr_matrix
from imblearn.ensemble import EasyEnsembleClassifier

from scipy.sparse import hstack
import regex as re
import re as r
import regex, string
import time
start = time.time()
class_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

train = pd.read_csv('../input/train.csv').fillna(' ')
test = pd.read_csv('../input/test.csv').fillna(' ')
repl = {
    "yay!": " good ",
    "yay": " good ",
    "yaay": " good ",
    "yaaay": " good ",
    "yaaaay": " good ",
    "yaaaaay": " good ",
    ":/": " bad ",
    ":&gt;": " sad ",
    ":')": " sad ",
    ":-(": " frown ",
    ":(": " frown ",
    ":s": " frown ",
    ":-s": " frown ",
    "&lt;3": " heart ",
    ":d": " smile ",
    ":p": " smile ",
    ":dd": " smile ",
    "8)": " smile ",
    ":-)": " smile ",
    ":)": " smile ",
    ";)": " smile ",
    "(-:": " smile ",
    "(:": " smile ",
    ":/": " worry ",
    ":&gt;": " angry ",
    ":')": " sad ",
    ":-(": " sad ",
    ":(": " sad ",
    ":s": " sad ",
    ":-s": " sad ",
    r"\br\b": "are",
    r"\bu\b": "you",
    r"\bhaha\b": "ha",
    r"\bhahaha\b": "ha",
    r"\bdon't\b": "do not",
    r"\bdoesn't\b": "does not",
    r"\bdidn't\b": "did not",
    r"\bhasn't\b": "has not",
    r"\bhaven't\b": "have not",
    r"\bhadn't\b": "had not",
    r"\bwon't\b": "will not",
    r"\bwouldn't\b": "would not",
    r"\bcan't\b": "can not",
    r"\bcannot\b": "can not",
    r"\bi'm\b": "i am",
    "m": "am",
    "r": "are",
    "u": "you",
    "haha": "ha",
    "hahaha": "ha",
    "don't": "do not",
    "doesn't": "does not",
    "didn't": "did not",
    "hasn't": "has not",
    "haven't": "have not",
    "hadn't": "had not",
    "won't": "will not",
    "wouldn't": "would not",
    "can't": "can not",
    "cannot": "can not",
    "i'm": "i am",
    "m": "am",
    "i'll" : "i will",
    "its" : "it is",
    "it's" : "it is",
    "'s" : " is",
    "that's" : "that is",
    "weren't" : "were not",
}

keys = [i for i in repl.keys()]

new_train_data = []
new_test_data = []
ltr = train["comment_text"].tolist()
lte = test["comment_text"].tolist()
for i in ltr:
    arr = str(i).split()
    xx = ""
    for j in arr:
        j = str(j).lower()
        if j[:4] == 'http' or j[:3] == 'www':
            continue
        if j in keys:
            # print("inn")
            j = repl[j]
        xx += j + " "
    new_train_data.append(xx)
for i in lte:
    arr = str(i).split()
    xx = ""
    for j in arr:
        j = str(j).lower()
        if j[:4] == 'http' or j[:3] == 'www':
            continue
        if j in keys:
            # print("inn")
            j = repl[j]
        xx += j + " "
    new_test_data.append(xx)
train["new_comment_text"] = new_train_data
test["new_comment_text"] = new_test_data

trate = train["new_comment_text"].tolist()
tete = test["new_comment_text"].tolist()
for i, c in enumerate(trate):
    trate[i] = re.sub('[^a-zA-Z ?!]+', '', str(trate[i]).lower())
for i, c in enumerate(tete):
    tete[i] = re.sub('[^a-zA-Z ?!]+', '', tete[i])
train["comment_text"] = trate
test["comment_text"] = tete
del trate, tete
train.drop(["new_comment_text"], axis=1, inplace=True)
test.drop(["new_comment_text"], axis=1, inplace=True)
train_text = train['comment_text']
test_text = test['comment_text']
all_text = pd.concat([train_text, test_text])
def prepare_for_char_n_gram(text):
    """ Simple text clean up process"""
    # 1. Go to lower case (only good for english)
    # Go to bytes_strings as I had issues removing all \n in r""
    clean = bytes(text.lower().encode('utf-8'))
    # 2. Drop \n and  \t
    clean = clean.replace(b"\n", b" ")
    clean = clean.replace(b"\t", b" ")
    clean = clean.replace(b"\b", b" ")
    clean = clean.replace(b"\r", b" ")
    
    exclude = r.compile(b'[%s]' % re.escape(bytes(string.punctuation.encode('utf-8'))))
    clean = b" ".join([exclude.sub(b'', token) for token in clean.split()])

    clean = r.sub(b"\d+", b" ", clean)
    return str(clean).encode('utf-8')

def count_regexp_occ(regexp="", text=None):
    """ Simple way to get the number of occurence of a regex"""
    return len(re.findall(regexp, text))
def get_indicators_and_clean_comments(df):
    df["clean_comment"] = df["comment_text"].apply(lambda x: prepare_for_char_n_gram(x))
for df in [train, test]:
   get_indicators_and_clean_comments(df)
train_text = train['clean_comment']
test_text = test['clean_comment']
all_text = pd.concat([train_text, test_text])
word_vectorizer = TfidfVectorizer(
        sublinear_tf=True,
        strip_accents='unicode',
        tokenizer=lambda x: regex.findall(r'[^\p{P}\W]+', x),
        analyzer='word',
        token_pattern='(?u)\\b\\w\\w+\\b\\w{,1}',
        min_df=5,
        norm='l2',
        ngram_range=(1, 1),
        max_features=31000)

word_vectorizer.fit(all_text)
train_word_features = word_vectorizer.transform(train_text)
test_word_features = word_vectorizer.transform(test_text)

char_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='char',
    token_pattern=None,
    min_df=5,
    norm='l2',
    ngram_range=(2, 4),
    max_features=23000)

char_vectorizer.fit(all_text)
train_char_features = char_vectorizer.transform(train_text)
test_char_features = char_vectorizer.transform(test_text)
train_features = hstack([train_word_features, train_char_features]).tocsr()
del train_word_features
test_features = hstack([test_word_features, test_char_features]).tocsr()
del test_word_features
del all_text
del test_text
del train_text
print(train_features.shape)
print(test_features.shape)
def pr(y_i, y):
    p = train_features[y==y_i].sum(0)
    return (p+1) / ((y==y_i).sum()+1)
scores = []
submission = pd.DataFrame.from_dict({'id': test['id']})
for class_name in class_names:
    train_target = train[class_name]
    y = train_target.values
    r = np.log(pr(1,y) / pr(0,y))
    x_nb = train_features.multiply(r)
    l = EasyEnsembleClassifier(base_estimator=LogisticRegression(C=2, solver='sag', max_iter=500))
    n = EasyEnsembleClassifier(base_estimator=SGDClassifier(alpha=.0002, max_iter=180, penalty="l2", loss='modified_huber'))
    o = LogisticRegression(C=2, dual=True, max_iter=500)
    p = RandomForestClassifier(criterion='gini',
            max_depth=100, max_features=1000, max_leaf_nodes=None, min_samples_split=10,
            min_weight_fraction_leaf=0.0, n_estimators=60)  
    m = VotingClassifier(estimators=[ ('lr', l), ('sgd', n),('lr1',o),('rdf',p)], voting='soft', weights=[0.94,1.35,0.58,0.9])
    m.fit(x_nb, y)
    """For cross validation scores please uncomment the following lines of code"""
    
#     cv_score = np.mean(cross_val_score(
#         m, x_nb, train_target, cv=5, scoring='roc_auc'))
#     scores.append(cv_score)
#     print('CV score for class {} is {}'.format(class_name, cv_score))
# print('Total CV score is {}'.format(np.mean(scores)))
    submission[class_name] = m.predict_proba(test_features.multiply(r))[:, 1]


submission.to_csv('submission.csv', index=False)
end=time.time()
print(end-start)