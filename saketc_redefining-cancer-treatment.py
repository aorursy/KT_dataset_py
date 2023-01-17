from sklearn import preprocessing, pipeline, feature_extraction, decomposition, model_selection, metrics, cross_validation, svm

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, BaggingClassifier

from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.feature_selection import SelectKBest, chi2

from sklearn.preprocessing import normalize, Imputer

from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import SGDClassifier

from sklearn.naive_bayes import MultinomialNB



import sklearn

import pandas as pd

import numpy as np

import xgboost as xgb



import datetime
train = pd.read_csv('../input/training_variants')

test = pd.read_csv('../input/test_variants')

trainx = pd.read_csv('../input/training_text', sep="\|\|", engine='python', header=None, skiprows=1, names=["ID","Text"])

testx = pd.read_csv('../input/test_text', sep="\|\|", engine='python', header=None, skiprows=1, names=["ID","Text"])



train = pd.merge(train, trainx, how='left', on='ID').fillna('')

y = train['Class'].values

train = train.drop(['Class'], axis=1)



test = pd.merge(test, testx, how='left', on='ID').fillna('')

pid = test['ID'].values
class cust_regression_vals(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):

    def fit(self, x, y=None):

        return self

    def transform(self, x):

        x = x.drop(['Gene', 'Variation','ID','Text'],axis=1).values

        return x



class cust_txt_col(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):

    def __init__(self, key):

        self.key = key

    def fit(self, x, y=None):

        return self

    def transform(self, x):

        return x[self.key].apply(str)



#commented for Kaggle Limits

('pi3', pipeline.Pipeline([('Text', cust_txt_col('Text')), 

                           ('tfidf_Text', feature_extraction.text.TfidfVectorizer(ngram_range=(1, 2))), 

                           ('tsvd3', decomposition.TruncatedSVD(n_components=50, n_iter=25, random_state=12))]))
print('Pipeline...')

fp = pipeline.Pipeline([

    ('union', pipeline.FeatureUnion(

        n_jobs = -1,

        transformer_list = [

            ('standard', cust_regression_vals()),

            ('pi1', pipeline.Pipeline([('Gene', cust_txt_col('Gene')), 

                                       ('count_Gene', feature_extraction.text.CountVectorizer(analyzer=u'char', ngram_range=(1, 8))), 

                                       ('tsvd1', decomposition.TruncatedSVD(n_components=20, n_iter=25, random_state=12))])),

            ('pi2', pipeline.Pipeline([('Variation', cust_txt_col('Variation')), 

                                       ('count_Variation', feature_extraction.text.CountVectorizer(analyzer=u'char', ngram_range=(1, 8))), 

                                       ('tsvd2', decomposition.TruncatedSVD(n_components=20, n_iter=25, random_state=12))])),

            #commented for Kaggle Limits

#             ('pi3', pipeline.Pipeline([('Text', cust_txt_col('Text')), 

#                                        ('hv', feature_extraction.text.HashingVectorizer(decode_error='ignore', n_features=2 ** 16, non_negative=True, ngram_range=(1, 3))),

#                                        ('tfidf_Text', feature_extraction.text.TfidfTransformer()), 

#                                        ('tsvd3', decomposition.TruncatedSVD(n_components=300, n_iter=25, random_state=12))]))



        

        ])

    )])





train = fp.fit_transform(train)

print (train.shape)



test_t = np.empty([0, train.shape[1]])

step = 200

for i in range(0, len(test), step):

    step_end = i+step

    step_end = step_end if step_end < len(test) else len(test)

    _test = fp.transform(test.iloc[i:step_end])

    test_t = np.vstack((test_t, _test))

test = test_t

print (test.shape)
y = y - 1 #fix for zero bound array



denom = 0

fold = 1 #Change to 5, 1 for Kaggle Limits

for i in range(fold):

    params = {

        'eta': 0.03333,

        'max_depth': 4,

        'objective': 'multi:softprob',

        'eval_metric': 'mlogloss',

        'num_class': 9,

        'seed': i,

        'silent': True

    }

    x1, x2, y1, y2 = model_selection.train_test_split(train, y, test_size=0.18, random_state=i)

    watchlist = [(xgb.DMatrix(x1, y1), 'train'), (xgb.DMatrix(x2, y2), 'valid')]

    model = xgb.train(params, xgb.DMatrix(x1, y1), 1000,  watchlist, verbose_eval=50, early_stopping_rounds=100)

    score1 = metrics.log_loss(y2, model.predict(xgb.DMatrix(x2), ntree_limit=model.best_ntree_limit), labels = list(range(9)))

    print(score1)

    #if score < 0.9:

    if denom != 0:

        pred = model.predict(xgb.DMatrix(test), ntree_limit=model.best_ntree_limit+80)

        preds += pred

    else:

        pred = model.predict(xgb.DMatrix(test), ntree_limit=model.best_ntree_limit+80)

        preds = pred.copy()

    denom += 1

    submission = pd.DataFrame(pred, columns=['class'+str(c+1) for c in range(9)])

    submission['ID'] = pid

    submission.to_csv('submission_xgb_fold_'  + str(i) + '.csv', index=False)

preds /= denom

submission = pd.DataFrame(preds, columns=['class'+str(c+1) for c in range(9)])

submission['ID'] = pid

submission.to_csv('submission_1_xgb.csv', index=False)