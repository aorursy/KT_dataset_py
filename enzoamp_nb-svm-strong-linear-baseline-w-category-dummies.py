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
from pathlib import Path

from sklearn.linear_model import LinearRegression, LogisticRegression

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
train = Path("/kaggle/input/google-quest-challenge/train.csv")

test =  Path("/kaggle/input/google-quest-challenge/test.csv")

sample_sub = Path("/kaggle/input/google-quest-challenge/sample_submission.csv")
train_df = pd.read_csv(train)

test_df = pd.read_csv(test)
train_df.shape, test_df.shape
train_targets = train_df.iloc[:, 11:]

train_feats = train_df.iloc[:, 1:11]



test_targets = test_df.iloc[:, 11:]

test_feats = test_df.iloc[:, 1:11]
train_targets.columns, train_feats.columns
question_target_cols = [col for col in train_targets.columns if col.split('_')[0] == 'question']

answer_target_cols = [col for col in train_targets.columns if col.split('_')[0] == 'answer']

question_feat_cols = [col for col in train_feats.columns if col.split('_')[0] == 'question']

answer_feat_cols = [col for col in train_feats.columns if col.split('_')[0] == 'answer']
train_df.head()
train_df.columns
print('Question:')

train_df.iloc[0].question_title, train_df.iloc[0].question_body
print('Answer:')

train_df.iloc[0].answer
def trunc_text(text, n=102):

    tokens = text.split()

    return ' '.join(tokens[: n]) if len(tokens) > n else text
train_question_text = train_feats.question_title + ' ' + train_feats.question_body

test_question_text = test_feats.question_title + ' ' + test_feats.question_body



train_answer_text = train_feats.answer

test_answer_text = test_feats.answer



lens = train_question_text.apply(lambda x: len(x.split()))

print(lens.describe())



train_question_text = train_question_text.apply(lambda x: trunc_text(x))

test_question_text = test_question_text.apply(lambda x: trunc_text(x))
train_targets['none_q'] = 1-train_targets[question_target_cols].max(axis=1)

train_targets.describe()
train_question_text.fillna("unknown", inplace=True)

test_question_text.fillna("unknown", inplace=True)
import re, string

re_tok = re.compile(f'([{string.punctuation}“”¨«»®´·º½¾¿¡§£₤‘’])')

def tokenize(s): return re_tok.sub(r' \1 ', s).split()
n = train_df.shape[0]

vec = TfidfVectorizer(ngram_range=(1,2), tokenizer=tokenize,

               min_df=3, max_df=0.9, strip_accents='unicode', use_idf=1,

               smooth_idf=1, sublinear_tf=1 )

trn_term_doc = vec.fit_transform(train_question_text)

test_term_doc = vec.transform(test_question_text)
vec_answers = TfidfVectorizer(ngram_range=(1,2), tokenizer=tokenize,

               min_df=3, max_df=0.9, strip_accents='unicode', use_idf=1,

               smooth_idf=1, sublinear_tf=1 )

trn_answers_term_doc = vec_answers.fit_transform(train_answer_text)

test_answers_term_doc = vec_answers.transform(test_answer_text)
trn_term_doc, test_term_doc
train_cat_dummies = pd.get_dummies(train_feats['category']).values

test_cat_dummies = pd.get_dummies(test_feats['category']).values
def pr(y_i, y):

    p = x[y==y_i].sum(0)

    return (p+1) / ((y==y_i).sum()+1)
def get_mdl(y):

    # Binarizing

    y = y.gt(0.5).astype(int)

    y = y.values

    r = np.log(pr(1,y) / pr(0,y))

    #m = LinearRegression()

    m = LogisticRegression(C=4, dual=True)

    x_nb = x.multiply(r)

    x_nb_cat = np.concatenate([x_nb.toarray(), train_cat_dummies], axis=1)

    return m.fit(x_nb_cat, y), r
x = trn_term_doc

test_x = test_term_doc



question_preds = np.zeros((test_df.shape[0], len(question_target_cols)))



for i, j in enumerate(question_target_cols):

    print('fitting', j)

    m,r = get_mdl(train_targets[j])

    test_x_nb_cat = np.concatenate([test_x.multiply(r).toarray(), test_cat_dummies], axis=1)

    #question_preds[:,i] = np.clip(m.predict(), 0, 1)

    print('predicting ...')

    question_preds[:,i] = m.predict_proba(test_x_nb_cat)[:,1]

    

question_preds_df = pd.DataFrame(question_preds, columns=question_target_cols)
x = trn_answers_term_doc

test_x = test_answers_term_doc



answer_preds = np.zeros((test_df.shape[0], len(answer_target_cols)))



for i, j in enumerate(answer_target_cols):

    print('fit', j)

    m,r = get_mdl(train_targets[j])

    test_x_nb_cat = np.concatenate([test_x.multiply(r).toarray(), test_cat_dummies], axis=1)

    #answer_preds[:,i] = np.clip(m.predict(test_x.multiply(r)), 0, 1)

    print('predicting ...')

    answer_preds[:,i] = m.predict_proba(test_x_nb_cat)[:,1]

    

answer_preds_df = pd.DataFrame(answer_preds, columns=answer_target_cols)
preds_df = pd.concat([question_preds_df, answer_preds_df], axis=1)

preds_df['qa_id'] = test_df.qa_id
sub_df = pd.read_csv(sample_sub)

sub_df_columns = sub_df.columns.values.tolist()

sub_df = preds_df[sub_df_columns]
sub_df
sub_df.to_csv("submission.csv", index = False)