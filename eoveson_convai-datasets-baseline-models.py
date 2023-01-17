import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
toxic_cmt = pd.read_table('../input/conversationaidataset/toxicity_annotated_comments.tsv')

toxic_annot = pd.read_table('../input/conversationaidataset/toxicity_annotations.tsv')

aggr_cmt = pd.read_table('../input/conversationaidataset/aggression_annotated_comments.tsv')

aggr_annot = pd.read_table('../input/conversationaidataset/aggression_annotations.tsv')

attack_cmt = pd.read_table('../input/conversationaidataset/attack_annotated_comments.tsv')

attack_annot = pd.read_table('../input/conversationaidataset/attack_annotations.tsv')
def JoinAndSanitize(cmt, annot):

    df = cmt.set_index('rev_id').join(annot.groupby(['rev_id']).mean())

    df = Sanitize(df)

    return df
def Sanitize(df):

    comment = 'comment' if 'comment' in df else 'comment_text'

    df[comment] = df[comment].str.lower().str.replace('newline_token', ' ')

    df[comment] = df[comment].fillna('erikov')

    return df
toxic = JoinAndSanitize(toxic_cmt, toxic_annot)

attack = JoinAndSanitize(attack_cmt, attack_annot)

aggression = JoinAndSanitize(aggr_cmt, aggr_annot)
len(attack), len(aggression)
attack['comment'].equals(aggression['comment'])
attack['attack'].corr(aggression['aggression'])
toxic.head()

#attack.head()

#aggression.head()
from sklearn.feature_extraction.text import TfidfVectorizer



def Tfidfize(df):

    # can tweak these as desired

    max_vocab = 200000

    split = 0.1



    comment = 'comment' if 'comment' in df else 'comment_text'

    

    tfidfer = TfidfVectorizer(ngram_range=(1,2), max_features=max_vocab,

                   use_idf=1, stop_words='english',

                   smooth_idf=1, sublinear_tf=1 )

    tfidf = tfidfer.fit_transform(df[comment])



    return tfidf, tfidfer
X_toxic, tfidfer_toxic = Tfidfize(toxic)

y_toxic = toxic['toxicity'].values

X_attack, tfidfer_attack = Tfidfize(attack)

y_attack = attack['attack'].values

X_aggression, tfidfer_aggression = Tfidfize(aggression)

y_aggression = aggression['aggression'].values
from sklearn.linear_model import Ridge

from sklearn.model_selection import cross_val_score



ridge = Ridge()

mse_toxic = -cross_val_score(ridge, X_toxic, y_toxic, scoring='neg_mean_squared_error')

mse_attack = -cross_val_score(ridge, X_attack, y_attack, scoring='neg_mean_squared_error')

mse_aggression = -cross_val_score(ridge, X_aggression, y_aggression, scoring='neg_mean_squared_error')
mse_toxic.mean(), mse_attack.mean(), mse_aggression.mean()
model_toxic = ridge.fit(X_toxic, y_toxic)

model_attack = ridge.fit(X_attack, y_attack)

model_aggression = ridge.fit(X_aggression, y_aggression)
train_orig = pd.read_csv('../input/jigsaw-toxic-comment-classification-challenge/train.csv')

test_orig = pd.read_csv('../input/jigsaw-toxic-comment-classification-challenge/test.csv')
train_orig = Sanitize(train_orig)

test_orig = Sanitize(test_orig)
def TfidfAndPredict(tfidfer, model):

    tfidf_train = tfidfer.transform(train_orig['comment_text'])

    tfidf_test = tfidfer.transform(test_orig['comment_text'])

    train_scores = model.predict(tfidf_train)

    test_scores = model.predict(tfidf_test)

    

    return train_scores, test_scores
toxic_tr_scores, toxic_t_scores = TfidfAndPredict(tfidfer_toxic, model_toxic)
toxic_tr_scores.shape, toxic_t_scores.shape
attack_tr_scores, attack_t_scores = TfidfAndPredict(tfidfer_attack, model_attack)
attack_tr_scores.shape, attack_t_scores.shape
aggression_tr_scores, aggression_t_scores = TfidfAndPredict(tfidfer_aggression, model_aggression)
aggression_tr_scores.shape, aggression_t_scores.shape
# toxic_level, to not be confused with original label 'toxic'

train_orig['toxic_level'] = toxic_tr_scores

train_orig['attack'] = attack_tr_scores

train_orig['aggression'] = aggression_tr_scores

test_orig['toxic_level'] = toxic_t_scores

test_orig['attack'] = attack_t_scores

test_orig['aggression'] = aggression_t_scores

train_orig.to_csv('train_with_convai.csv', index=False)

test_orig.to_csv('test_with_convai.csv', index=False)