import pandas as pd, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import (train_test_split,  RandomizedSearchCV)
from nltk.corpus import stopwords
import nltk
import re
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
subm = pd.read_csv('../input/sample_submission.csv')

#flag to split training to collect a hold-out set
SPLIT_TRAINING = False
if SPLIT_TRAINING:
    del test
    train, test = train_test_split( train, test_size=.8)
    train = train.reset_index()
    test = test.reset_index()
label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
train['none'] = 1-train[label_cols].max(axis=1)
COMMENT = 'comment_text'
train[COMMENT].fillna("unknown", inplace=True)
test[COMMENT].fillna("unknown", inplace=True)
#some formatting
stop_words_en = set(stopwords.words("english"))
word_pattern = "\w[a-z-_]+\w"

def normalize(s):
    """
    Given a text, cleans and normalizes it. Feel free to add your own stuff.
    """
    s = s.lower()
    # Replace ips
    #s = re.sub(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', ' _ip_ ', s)
    s = re.sub(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', '', s)
    # Isolate punctuation
    s = re.sub(r'([\'\"\.\(\)\!\?\-\\\/\,])', r' \1 ', s)
    # Remove some special characters
    s = re.sub(r'([\;\:\|•«\n])', ' ', s)
    # Replace numbers and symbols with language
    s = s.replace('&', ' and ')
    s = s.replace('@', ' at ')
    s = s.replace('0', ' zero ')
    s = s.replace('1', ' one ')
    s = s.replace('2', ' two ')
    s = s.replace('3', ' three ')
    s = s.replace('4', ' four ')
    s = s.replace('5', ' five ')
    s = s.replace('6', ' six ')
    s = s.replace('7', ' seven ')
    s = s.replace('8', ' eight ')
    s = s.replace('9', ' nine ')
    s = s.replace("!", " exclamationmark ")
    s = s.replace("?", " qestionmark ")
    
    #remove english stopwords -- shown to decrease the performance
    #s = " ".join( word for word in re.findall(word_pattern,s) if word not in stop_words_en)
    return s

train[COMMENT] = train[COMMENT].map( normalize)
test[COMMENT] = test[COMMENT].map( normalize)
import re, string
re_tok = re.compile(f'([{string.punctuation}“”¨«»®´·º½¾¿¡§£₤‘’])')
def tokenize(s): return re_tok.sub(r' \1 ', s).split()
n = train.shape[0]

vec = TfidfVectorizer(tokenizer=tokenize,
                      min_df=2,
                      max_df=.625,
                      strip_accents='unicode',
                      use_idf=1,
                      smooth_idf=True,
                      sublinear_tf=True)#, stop_words=stop_words_en)
trn_term_doc = vec.fit_transform( train[COMMENT])
test_term_doc = vec.transform( test[COMMENT])
def pr(y_i, y):
    p = x[y==y_i].sum(0)
    return (p+1) / ((y==y_i).sum()+1)
x = trn_term_doc
test_x = test_term_doc
class random_log_uniform:
    def __init__ (self, power_low, power_high):
        self.power_low = power_low
        self.power_high = power_high
    def rvs(self, random_state=0):
        rnd_power = np.random.rand() * (self.power_high - self.power_low ) + self.power_low
        return 10**rnd_power
    
def get_mdl_training(y, parameters):
    y = y.values
    r = np.log(pr(1,y) / pr(0,y))
    model = LogisticRegression( dual=True)
    params = { "C": random_log_uniform( np.log10(0.01), np.log10(10))}
    x_nb = x.multiply(r)
    model_grid = RandomizedSearchCV( model, params, cv=5, n_jobs=1, n_iter=50, scoring='roc_auc')
    model_grid.fit(x_nb,y)
    m = model_grid.best_estimator_
    parameters.append( model_grid.best_params_["C"])
    return m.fit(x_nb, y), r

def get_mdl(y,C):
    y = y.values
    r = np.log(pr(1,y) / pr(0,y))
    m = LogisticRegression(C=C, dual=True)
    x_nb = x.multiply(r)
    return m.fit(x_nb, y), r
preds = np.zeros((len(test), len(label_cols)))

#last cv-fitted hyperparameters
Cvalues = [1.391737250242211, 0.17361472592588692, 0.45345054479480851,
           0.58227213530514976, 0.59166916356898391, 0.34278082413675537]

TRAINING_HYPERPARAMETERS = False
if TRAINING_HYPERPARAMETERS:
    Cvalues = []
for i, j in enumerate(label_cols):
    print('fitting %s ..' %j)
    if TRAINING_HYPERPARAMETERS:
        m,r = get_mdl_training(train[j], Cvalues)
    else:
        m,r = get_mdl(train[j], Cvalues[i])
    preds[:,i] = m.predict_proba(test_x.multiply(r))[:,1]
if TRAINING_HYPERPARAMETERS:
    print( "\tfitted coefficients: ", Cvalues)
submid = pd.DataFrame({'id': subm["id"]})
submission = pd.concat([submid, pd.DataFrame(preds, columns = label_cols)], axis=1)
submission.to_csv('submission.csv', index=False)
if SPLIT_TRAINING:
    from sklearn.metrics import roc_auc_score
    for index, _ in enumerate( label_cols):
        print( "%.5f" % roc_auc_score( test[label_cols].values[:,index], preds[:,index]), end=" ")
    print( "\n%.5f" %roc_auc_score( test[label_cols].values, preds))