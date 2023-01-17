import numpy as np
import pandas as pd
import os
import xgboost as xgb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
import scipy as sp
from sklearn import linear_model
from functools import partial
from sklearn import metrics
from collections import Counter
import json
import lightgbm as lgb
# The following 3 functions have been taken from Ben Hamner's github repository
# https://github.com/benhamner/Metrics
def confusion_matrix(rater_a, rater_b, min_rating=None, max_rating=None):
    """
    Returns the confusion matrix between rater's ratings
    """
    assert(len(rater_a) == len(rater_b))
    if min_rating is None:
        min_rating = min(rater_a + rater_b)
    if max_rating is None:
        max_rating = max(rater_a + rater_b)
    num_ratings = int(max_rating - min_rating + 1)
    conf_mat = [[0 for i in range(num_ratings)]
                for j in range(num_ratings)]
    for a, b in zip(rater_a, rater_b):
        conf_mat[a - min_rating][b - min_rating] += 1
    return conf_mat


def histogram(ratings, min_rating=None, max_rating=None):
    """
    Returns the counts of each type of rating that a rater made
    """
    if min_rating is None:
        min_rating = min(ratings)
    if max_rating is None:
        max_rating = max(ratings)
    num_ratings = int(max_rating - min_rating + 1)
    hist_ratings = [0 for x in range(num_ratings)]
    for r in ratings:
        hist_ratings[r - min_rating] += 1
    return hist_ratings


def quadratic_weighted_kappa(y, y_pred):
    """
    Calculates the quadratic weighted kappa
    axquadratic_weighted_kappa calculates the quadratic weighted kappa
    value, which is a measure of inter-rater agreement between two raters
    that provide discrete numeric ratings.  Potential values range from -1
    (representing complete disagreement) to 1 (representing complete
    agreement).  A kappa value of 0 is expected if all agreement is due to
    chance.
    quadratic_weighted_kappa(rater_a, rater_b), where rater_a and rater_b
    each correspond to a list of integer ratings.  These lists must have the
    same length.
    The ratings should be integers, and it is assumed that they contain
    the complete range of possible ratings.
    quadratic_weighted_kappa(X, min_rating, max_rating), where min_rating
    is the minimum possible rating, and max_rating is the maximum possible
    rating
    """
    rater_a = y
    rater_b = y_pred
    min_rating=None
    max_rating=None
    rater_a = np.array(rater_a, dtype=int)
    rater_b = np.array(rater_b, dtype=int)
    assert(len(rater_a) == len(rater_b))
    if min_rating is None:
        min_rating = min(min(rater_a), min(rater_b))
    if max_rating is None:
        max_rating = max(max(rater_a), max(rater_b))
    conf_mat = confusion_matrix(rater_a, rater_b,
                                min_rating, max_rating)
    num_ratings = len(conf_mat)
    num_scored_items = float(len(rater_a))

    hist_rater_a = histogram(rater_a, min_rating, max_rating)
    hist_rater_b = histogram(rater_b, min_rating, max_rating)

    numerator = 0.0
    denominator = 0.0

    for i in range(num_ratings):
        for j in range(num_ratings):
            expected_count = (hist_rater_a[i] * hist_rater_b[j]
                              / num_scored_items)
            d = pow(i - j, 2.0) / pow(num_ratings - 1, 2.0)
            numerator += d * conf_mat[i][j] / num_scored_items
            denominator += d * expected_count / num_scored_items

    return (1.0 - numerator / denominator)
class OptimizedRounder(object):
    def __init__(self):
        self.coef_ = 0

    def _kappa_loss(self, coef, X, y):
        X_p = np.copy(X)
        for i, pred in enumerate(X_p):
            if pred < coef[0]:
                X_p[i] = 0
            elif pred >= coef[0] and pred < coef[1]:
                X_p[i] = 1
            elif pred >= coef[1] and pred < coef[2]:
                X_p[i] = 2
            elif pred >= coef[2] and pred < coef[3]:
                X_p[i] = 3
            else:
                X_p[i] = 4

        ll = quadratic_weighted_kappa(y, X_p)
        return -ll

    def fit(self, X, y):
        loss_partial = partial(self._kappa_loss, X=X, y=y)
        initial_coef = [0.5, 1.5, 2.5, 3.5]
        self.coef_ = sp.optimize.minimize(loss_partial, initial_coef, method='nelder-mead')

    def predict(self, X, coef):
        X_p = np.copy(X)
        for i, pred in enumerate(X_p):
            if pred < coef[0]:
                X_p[i] = 0
            elif pred >= coef[0] and pred < coef[1]:
                X_p[i] = 1
            elif pred >= coef[1] and pred < coef[2]:
                X_p[i] = 2
            elif pred >= coef[2] and pred < coef[3]:
                X_p[i] = 3
            else:
                X_p[i] = 4
        return X_p

    def coefficients(self):
        return self.coef_['x']
train = pd.read_csv('../input/train/train.csv')
test = pd.read_csv('../input/test/test.csv')
# train[train.AdoptionSpeed==0]
doc_sent_mag = []
doc_sent_score = []
nf_count = 0
for pet in train.PetID.values:
    try:
        with open('../input/train_sentiment/' + pet + '.json', 'r') as f:
            sentiment = json.load(f)
        doc_sent_mag.append(sentiment['documentSentiment']['magnitude'])
        doc_sent_score.append(sentiment['documentSentiment']['score'])
    except FileNotFoundError:
        nf_count += 1
        doc_sent_mag.append(-1)
        doc_sent_score.append(-1)
train['doc_sent_mag'] = doc_sent_mag
train['doc_sent_score'] = doc_sent_score
nf_count
doc_sent_mag = []
doc_sent_score = []
nf_count = 0
for pet in test.PetID.values:
    try:
        with open('../input/test_sentiment/' + pet + '.json', 'r') as f:
            sentiment = json.load(f)
        doc_sent_mag.append(sentiment['documentSentiment']['magnitude'])
        doc_sent_score.append(sentiment['documentSentiment']['score'])
    except FileNotFoundError:
        nf_count += 1
        doc_sent_mag.append(-1)
        doc_sent_score.append(-1)
test['doc_sent_mag'] = doc_sent_mag
test['doc_sent_score'] = doc_sent_score
nf_count
lbl_enc = LabelEncoder()
lbl_enc.fit(train.RescuerID.values.tolist() + test.RescuerID.values.tolist())
train.RescuerID = lbl_enc.transform(train.RescuerID.values)
test.RescuerID = lbl_enc.transform(test.RescuerID.values)
train_desc = train.Description.fillna("none").values
test_desc = test.Description.fillna("none").values

tfv = TfidfVectorizer(min_df=3,  max_features=None,
        strip_accents='unicode', analyzer='word', token_pattern=r'\w{1,}',
        ngram_range=(1, 3), use_idf=1, smooth_idf=1, sublinear_tf=1,
        stop_words = 'english')
    
# Fit TFIDF
tfv.fit(list(train_desc) + list(test_desc))
X =  tfv.transform(train_desc)
X_test = tfv.transform(test_desc)


svd = TruncatedSVD(n_components=180)
svd.fit(X)
X = svd.transform(X)
X_test = svd.transform(X_test)
y = train.AdoptionSpeed
y.value_counts()
train = np.hstack((train.drop(['Name', 'Description', 'PetID', 'AdoptionSpeed'], axis=1).values, X))
test = np.hstack((test.drop(['Name', 'Description', 'PetID'], axis=1).values, X_test))
train_predictions = np.zeros((train.shape[0], 1))
test_predictions = np.zeros((test.shape[0], 1))
zero_test_predictions = np.zeros((test.shape[0], 1))
FOLDS = 3

print("stratified k-folds")
skf = StratifiedKFold(n_splits=FOLDS, random_state=42, shuffle=True)
skf.get_n_splits(train, y)
cv_scores = []
fold = 1
coefficients = np.zeros((FOLDS, 4))
for train_idx, valid_idx in skf.split(train, y):
    xtrain, xvalid = train[train_idx], train[valid_idx]
    xtrain_text, xvalid_text = X[train_idx], X[valid_idx]
    ytrain, yvalid = y.iloc[train_idx], y.iloc[valid_idx]
    
    w = y.value_counts()
    weights = {i : np.sum(w) / w[i] for i in w.index}
    print(weights)
    
    #model = xgb.XGBRegressor(n_estimators=500, nthread=-1, max_depth=19, learning_rate=0.01, min_child_weight = 150, colsample_bytree=0.8)
    lgb_params = {
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'learning_rate': 0.005,
    'subsample': .8,
    'colsample_bytree': 0.8,
    'min_split_gain': 0.006,
    'min_child_samples': 150,
    'min_child_weight': 0.1,
    'max_depth': 17,
    'n_estimators': 10000,
    'num_leaves': 80,
    'silent': -1,
    'verbose': -1,
    'max_depth': 11,
    'random_state': 2018
    }
    
    model = lgb.LGBMRegressor(**lgb_params)
    model.fit(
        xtrain, ytrain,
        eval_set=[(xvalid, yvalid)],
        eval_metric='rmse',
        verbose=100,
        early_stopping_rounds=100
    )
    
    #model.fit(xtrain, ytrain)
    valid_preds = model.predict(xvalid, num_iteration=model.best_iteration_)

    optR = OptimizedRounder()
    optR.fit(valid_preds, yvalid.values)
    coefficients[fold-1,:] = optR.coefficients()
    valid_p = optR.predict(valid_preds, coefficients[fold-1,:])

    print("Valid Counts = ", Counter(yvalid.values))
    print("Predicted Counts = ", Counter(valid_p))
    
    test_preds = model.predict(test, num_iteration=model.best_iteration_)

    scr = quadratic_weighted_kappa(yvalid.values, valid_p)
    cv_scores.append(scr)
    print("Fold = {}. QWK = {}. Coef = {}".format(fold, scr, coefficients[fold-1,:]))
    print("\n")
    train_predictions[valid_idx] = valid_preds.reshape(-1, 1)
    test_predictions += test_preds.reshape(-1, 1)
    fold += 1
test_predictions = test_predictions * 1./FOLDS
print("Mean Score: {}. Std Dev: {}. Mean Coeff: {}".format(np.mean(cv_scores), np.std(cv_scores), np.mean(coefficients, axis=0)))

optR = OptimizedRounder()
train_predictions = np.array([item for sublist in train_predictions for item in sublist])
optR.fit(train_predictions, y)
coefficients = optR.coefficients()
print(quadratic_weighted_kappa(y, optR.predict(train_predictions, coefficients)))
predictions = optR.predict(test_predictions, coefficients).astype(int)
predictions = [item for sublist in predictions for item in sublist]
sample = pd.read_csv('../input/test/sample_submission.csv')
sample.AdoptionSpeed = predictions
sample.to_csv('submission.csv', index=False)
sample.dtypes
sample.AdoptionSpeed.value_counts()
sample.head()

