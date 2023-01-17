import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir("../input"))
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
sub= pd.read_csv('../input/sample_submission.csv')
print("Train shape : ",train.shape)
print("Test shape : ",test.shape)
print("sub : ", sub.shape)
import os
import time
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from tqdm import tqdm
import math
from sklearn.model_selection import train_test_split
from sklearn import metrics
## split to train and val
train, val = train_test_split(train, test_size=0.08, random_state=2018)
train.head()
train['question_text'][0]
lens = train.question_text.str.len()
lens.mean(), lens.std(), lens.max()
from matplotlib import pyplot as plt
%matplotlib inline
lens.hist();
plt.title('Counts for different length of questions')
plt.ylabel('Count')
plt.xlabel('Length of questions')

train.loc[lens.argmax()]['question_text']
train.isnull().sum()
import re, string
re_tok = re.compile(f'([{string.punctuation}“”¨«»®´·º½¾¿¡§£₤‘’])')
def tokenize(s): return re_tok.sub(r' \1 ', s).split()
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
n = train.shape[0]
vec = TfidfVectorizer(ngram_range=(1,2), tokenizer=tokenize,
               min_df=3, max_df=0.9, strip_accents='unicode', use_idf=1,
               smooth_idf=1, sublinear_tf=1 )
trn_term_doc = vec.fit_transform(train['question_text'])
val_term_doc = vec.transform(val['question_text'])
test_term_doc = vec.transform(test['question_text'])


vec_count = CountVectorizer(ngram_range=(1,2), tokenizer=tokenize,
              strip_accents='unicode' )
trn_term_doc_count = vec_count.fit_transform(train['question_text'])
word_dict = vec_count.vocabulary_
word_df = pd.DataFrame.from_dict(word_dict, orient='index')
word_df = word_df.rename({0:'Count'}, axis=1)
word_df.sort_values('Count').head(20)
trn_term_doc, val_term_doc, test_term_doc
def pr(y_i, y):
    p = x[y==y_i].sum(0)
    return (p+1) / ((y==y_i).sum()+1)

x = trn_term_doc
val_x = val_term_doc
test_x = test_term_doc
def get_mdl(y):
    y = y.values
    r = np.log(pr(1,y) / pr(0,y))
    m = LogisticRegression(C=4, dual=True)
    x_nb = x.multiply(r)
    return m.fit(x_nb, y), r
preds = np.zeros((len(val), 1))
m,r = get_mdl(train['target'])
preds[:,0] = m.predict_proba(val_x.multiply(r))[:,1]
preds.shape
val.loc[:, 'prediction'] = preds
from sklearn.metrics import f1_score
best_score = []
ts = np.arange(0, 1, 0.05)
for t in ts :
    preds = np.where(val['prediction'] > t, 1, 0)
    score =  f1_score(val['target'], preds )
    print('Threshold: ', t , ',f1 score :', score )
    best_score.append(score)
    
best_idx = np.array(best_score).argmax()
    
print('Best threshold :', ts[best_idx], 'Best Score :', best_score[best_idx] )
train = pd.read_csv("../input/train.csv")
m,r = get_mdl(train['target'])
preds = np.zeros((len(test), 1))
preds[:,0] = m.predict_proba(test_x.multiply(r))[:,1]
submission = pd.DataFrame({'qid': sub["qid"]})
submission['prediction'] = np.where(preds > ts[best_idx], 1 , 0)
submission.to_csv('submission.csv', index=False)
