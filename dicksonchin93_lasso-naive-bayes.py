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
import pandas as pd, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
train = pd.read_csv('../input/train.csv', usecols=['description', 'deal_probability'])
test = pd.read_csv('../input/test.csv', usecols=['description'])
COMMENT = 'description'
train[COMMENT].fillna("неизвестный", inplace=True)
test[COMMENT].fillna("неизвестный", inplace=True)
import re, string
re_tok = re.compile(f'([{string.punctuation}“”¨«»®´·º½¾¿¡§£₤‘’])')
def tokenize(s): return re_tok.sub(r' \1 ', s).split()
n = train.shape[0]
vec = TfidfVectorizer(ngram_range=(1,2), tokenizer=tokenize,
               min_df=3, max_df=0.9, strip_accents='unicode', use_idf=1,
               smooth_idf=1, sublinear_tf=1 )
trn_term_doc = vec.fit_transform(train[COMMENT])
test_term_doc = vec.transform(test[COMMENT])
trn_term_doc, test_term_doc

def pr(y_i, y, x_temp):
    p = x_temp[y==y_i].sum(0)
    return (p+1) / ((y==y_i).sum()+1)
x = trn_term_doc
test_x = test_term_doc
def get_mdl(y, x_temp):
    y = y.values
    r = np.log(pr(1,y, x_temp) / pr(0,y, x_temp))
    m = linear_model.Lasso(alpha=0.1)
    x_nb = x_temp.multiply(r)
    return m.fit(x_nb, y), r
from sklearn.model_selection import KFold
from sklearn import metrics
from sklearn import linear_model
from sklearn.isotonic import IsotonicRegression
RS = 20180601
folds = KFold(n_splits=5, shuffle=True, random_state=546789)
oof_preds = np.zeros(x.shape[0])
test_predicts_list = []
np.random.seed(RS)
for n_fold, (trn_idx, val_idx) in enumerate(folds.split(x)):
    print("fold {}".format(n_fold))
    trn_x, trn_y = x[trn_idx], train.loc[trn_idx]
    val_x, val_y = x[val_idx],  train.loc[val_idx]
    m,r = get_mdl(trn_y['deal_probability'], trn_x)
    oof = m.predict(val_x.multiply(r))
    oof_preds[val_idx] = oof
    print('RMSE:', np.sqrt(metrics.mean_squared_error(val_y['deal_probability'].values, oof)))
    preds = m.predict(test_x.multiply(r))
    test_predicts_list.append(preds)

test_predicts = np.ones(test_predicts_list[0].shape)
for fold_predict in test_predicts_list:
    test_predicts *= fold_predict

test_predicts **= (1. / len(test_predicts_list))
np.save('lasso_naivebayes_oof.npy', oof_preds)
np.save('lasso_naivebayes_preds.npy', test_predicts)
