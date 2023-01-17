import json



import scipy as sp

import pandas as pd

import numpy as np



from functools import partial

from math import sqrt



from sklearn.metrics import cohen_kappa_score, mean_squared_error

from sklearn.metrics import confusion_matrix as sk_cmatrix

from sklearn.model_selection import StratifiedKFold



from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.decomposition import TruncatedSVD



from collections import Counter



from catboost import CatBoostRegressor

np.random.seed(724)
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

    

def rmse(actual, predicted):

    return sqrt(mean_squared_error(actual, predicted))    
print('Train')

train = pd.read_csv("../input/train/train.csv")

print(train.shape)



print('Test')

test = pd.read_csv("../input/test/test.csv")

print(test.shape)



print('Breeds')

breeds = pd.read_csv("../input/breed_labels.csv")

print(breeds.shape)



print('Colors')

colors = pd.read_csv("../input/color_labels.csv")

print(colors.shape)



print('States')

states = pd.read_csv("../input/state_labels.csv")

print(states.shape)



target = train['AdoptionSpeed']

train_id = train['PetID']

test_id = test['PetID']

train.drop(['AdoptionSpeed', 'PetID'], axis=1, inplace=True)

test.drop(['PetID'], axis=1, inplace=True)



doc_sent_mag = []

doc_sent_score = []

nf_count = 0

for pet in train_id:

    try:

        with open('../input/train_sentiment/' + pet + '.json', 'r') as f:

            sentiment = json.load(f)

        doc_sent_mag.append(sentiment['documentSentiment']['magnitude'])

        doc_sent_score.append(sentiment['documentSentiment']['score'])

    except FileNotFoundError:

        nf_count += 1

        doc_sent_mag.append(-1)

        doc_sent_score.append(-1)



train.loc[:, 'doc_sent_mag'] = doc_sent_mag

train.loc[:, 'doc_sent_score'] = doc_sent_score



doc_sent_mag = []

doc_sent_score = []

nf_count = 0

for pet in test_id:

    try:

        with open('../input/test_sentiment/' + pet + '.json', 'r') as f:

            sentiment = json.load(f)

        doc_sent_mag.append(sentiment['documentSentiment']['magnitude'])

        doc_sent_score.append(sentiment['documentSentiment']['score'])

    except FileNotFoundError:

        nf_count += 1

        doc_sent_mag.append(-1)

        doc_sent_score.append(-1)



test.loc[:, 'doc_sent_mag'] = doc_sent_mag

test.loc[:, 'doc_sent_score'] = doc_sent_score
SVD_COMPONENTS = 120



train_desc = train.Description.fillna("none").values

test_desc = test.Description.fillna("none").values



tfv = TfidfVectorizer(min_df=3,  max_features=10000,

        strip_accents='unicode', analyzer='word', token_pattern=r'\w{1,}',

        ngram_range=(1, 3), use_idf=1, smooth_idf=1, sublinear_tf=1,

        stop_words = 'english')

    

# Fit TFIDF

tfv.fit(list(train_desc))

X =  tfv.transform(train_desc)

X_test = tfv.transform(test_desc)



svd = TruncatedSVD(n_components=SVD_COMPONENTS)

svd.fit(X)

X = svd.transform(X)

X = pd.DataFrame(X, columns=['svd_{}'.format(i) for i in range(SVD_COMPONENTS)])

train = pd.concat((train, X), axis=1)

X_test = svd.transform(X_test)

X_test = pd.DataFrame(X_test, columns=['svd_{}'.format(i) for i in range(SVD_COMPONENTS)])

test = pd.concat((test, X_test), axis=1)

vertex_xs = []

vertex_ys = []

bounding_confidences = []

bounding_importance_fracs = []

dominant_blues = []

dominant_greens = []

dominant_reds = []

dominant_pixel_fracs = []

dominant_scores = []

label_descriptions = []

label_scores = []

nf_count = 0

nl_count = 0

for pet in train_id:

    try:

        with open('../input/train_metadata/' + pet + '-1.json', 'r') as f:

            data = json.load(f)

        vertex_x = data['cropHintsAnnotation']['cropHints'][0]['boundingPoly']['vertices'][2]['x']

        vertex_xs.append(vertex_x)

        vertex_y = data['cropHintsAnnotation']['cropHints'][0]['boundingPoly']['vertices'][2]['y']

        vertex_ys.append(vertex_y)

        bounding_confidence = data['cropHintsAnnotation']['cropHints'][0]['confidence']

        bounding_confidences.append(bounding_confidence)

        bounding_importance_frac = data['cropHintsAnnotation']['cropHints'][0].get('importanceFraction', -1)

        bounding_importance_fracs.append(bounding_importance_frac)

        dominant_blue = data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['color']['blue']

        dominant_blues.append(dominant_blue)

        dominant_green = data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['color']['green']

        dominant_greens.append(dominant_green)

        dominant_red = data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['color']['red']

        dominant_reds.append(dominant_red)

        dominant_pixel_frac = data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['pixelFraction']

        dominant_pixel_fracs.append(dominant_pixel_frac)

        dominant_score = data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['score']

        dominant_scores.append(dominant_score)

        if data.get('labelAnnotations'):

            label_description = data['labelAnnotations'][0]['description']

            label_descriptions.append(label_description)

            label_score = data['labelAnnotations'][0]['score']

            label_scores.append(label_score)

        else:

            nl_count += 1

            label_descriptions.append('nothing')

            label_scores.append(-1)

    except FileNotFoundError:

        nf_count += 1

        vertex_xs.append(-1)

        vertex_ys.append(-1)

        bounding_confidences.append(-1)

        bounding_importance_fracs.append(-1)

        dominant_blues.append(-1)

        dominant_greens.append(-1)

        dominant_reds.append(-1)

        dominant_pixel_fracs.append(-1)

        dominant_scores.append(-1)

        label_descriptions.append('nothing')

        label_scores.append(-1)



print(nf_count)

print(nl_count)

train.loc[:, 'vertex_x'] = vertex_xs

train.loc[:, 'vertex_y'] = vertex_ys

train.loc[:, 'bounding_confidence'] = bounding_confidences

train.loc[:, 'bounding_importance'] = bounding_importance_fracs

train.loc[:, 'dominant_blue'] = dominant_blues

train.loc[:, 'dominant_green'] = dominant_greens

train.loc[:, 'dominant_red'] = dominant_reds

train.loc[:, 'dominant_pixel_frac'] = dominant_pixel_fracs

train.loc[:, 'dominant_score'] = dominant_scores

train.loc[:, 'label_description'] = label_descriptions

train.loc[:, 'label_score'] = label_scores





vertex_xs = []

vertex_ys = []

bounding_confidences = []

bounding_importance_fracs = []

dominant_blues = []

dominant_greens = []

dominant_reds = []

dominant_pixel_fracs = []

dominant_scores = []

label_descriptions = []

label_scores = []

nf_count = 0

nl_count = 0

for pet in test_id:

    try:

        with open('../input/test_metadata/' + pet + '-1.json', 'r') as f:

            data = json.load(f)

        vertex_x = data['cropHintsAnnotation']['cropHints'][0]['boundingPoly']['vertices'][2]['x']

        vertex_xs.append(vertex_x)

        vertex_y = data['cropHintsAnnotation']['cropHints'][0]['boundingPoly']['vertices'][2]['y']

        vertex_ys.append(vertex_y)

        bounding_confidence = data['cropHintsAnnotation']['cropHints'][0]['confidence']

        bounding_confidences.append(bounding_confidence)

        bounding_importance_frac = data['cropHintsAnnotation']['cropHints'][0].get('importanceFraction', -1)

        bounding_importance_fracs.append(bounding_importance_frac)

        dominant_blue = data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['color']['blue']

        dominant_blues.append(dominant_blue)

        dominant_green = data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['color']['green']

        dominant_greens.append(dominant_green)

        dominant_red = data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['color']['red']

        dominant_reds.append(dominant_red)

        dominant_pixel_frac = data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['pixelFraction']

        dominant_pixel_fracs.append(dominant_pixel_frac)

        dominant_score = data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['score']

        dominant_scores.append(dominant_score)

        if data.get('labelAnnotations'):

            label_description = data['labelAnnotations'][0]['description']

            label_descriptions.append(label_description)

            label_score = data['labelAnnotations'][0]['score']

            label_scores.append(label_score)

        else:

            nl_count += 1

            label_descriptions.append('nothing')

            label_scores.append(-1)

    except FileNotFoundError:

        nf_count += 1

        vertex_xs.append(-1)

        vertex_ys.append(-1)

        bounding_confidences.append(-1)

        bounding_importance_fracs.append(-1)

        dominant_blues.append(-1)

        dominant_greens.append(-1)

        dominant_reds.append(-1)

        dominant_pixel_fracs.append(-1)

        dominant_scores.append(-1)

        label_descriptions.append('nothing')

        label_scores.append(-1)



print(nf_count)

test.loc[:, 'vertex_x'] = vertex_xs

test.loc[:, 'vertex_y'] = vertex_ys

test.loc[:, 'bounding_confidence'] = bounding_confidences

test.loc[:, 'bounding_importance'] = bounding_importance_fracs

test.loc[:, 'dominant_blue'] = dominant_blues

test.loc[:, 'dominant_green'] = dominant_greens

test.loc[:, 'dominant_red'] = dominant_reds

test.loc[:, 'dominant_pixel_frac'] = dominant_pixel_fracs

test.loc[:, 'dominant_score'] = dominant_scores

test.loc[:, 'label_description'] = label_descriptions

test.loc[:, 'label_score'] = label_scores
train.drop(['Name', 'RescuerID', 'Description'], axis=1, inplace=True)

test.drop(['Name', 'RescuerID', 'Description'], axis=1, inplace=True)



numeric_cols = ['Age', 'Quantity', 'Fee', 'VideoAmt', 'PhotoAmt', 'AdoptionSpeed', 'doc_sent_mag', 'doc_sent_score', 'dominant_score', 'dominant_pixel_frac', 'dominant_red', 'dominant_green', 'dominant_blue', 'bounding_importance', 'bounding_confidence', 'vertex_x', 'vertex_y', 'label_score'] + ['svd_{}'.format(i) for i in range(SVD_COMPONENTS)]

cat_cols = list(set(train.columns) - set(numeric_cols))

train.loc[:, cat_cols] = train[cat_cols].astype('category')

test.loc[:, cat_cols] = test[cat_cols].astype('category')

print(train.shape)

print(test.shape)



# get the categorical features

foo = train.dtypes

cat_feature_names = foo[foo == "category"]

cat_features = [train.columns.get_loc(c) for c in train.columns if c in cat_feature_names]
N_SPLITS = 3



def run_cv_model(train, test, target, model_fn, params={}, eval_fn=None, label='model'):

    kf = StratifiedKFold(n_splits=N_SPLITS, random_state=2407, shuffle=True)

    fold_splits = kf.split(train, target)

    cv_scores = []

    qwk_scores = []

    pred_full_test = 0

    pred_train = np.zeros((train.shape[0], N_SPLITS))

    all_coefficients = np.zeros((N_SPLITS, 4))

    feature_importance_df = pd.DataFrame()

    i = 1

    for dev_index, val_index in fold_splits:

        print()

        print('Started ' + label + ' fold ' + str(i) + '/' + str(N_SPLITS))

        if isinstance(train, pd.DataFrame):

            dev_X, val_X = train.iloc[dev_index], train.iloc[val_index]

            dev_y, val_y = target[dev_index], target[val_index]

        else:

            dev_X, val_X = train[dev_index], train[val_index]

            dev_y, val_y = target[dev_index], target[val_index]

        params2 = params.copy()

        pred_val_y, pred_test_y, importances, coefficients, qwk = model_fn(dev_X, dev_y, val_X, val_y, test, params2)        

        pred_full_test = pred_full_test + pred_test_y

        

        pred_train[val_index] = pred_val_y

        all_coefficients[i-1, :] = coefficients

        if eval_fn is not None:

            cv_score = eval_fn(val_y, pred_val_y)

            cv_scores.append(cv_score)

            qwk_scores.append(qwk)

            print(label + ' cv score {}: RMSE {} QWK {}'.format(i, cv_score, qwk))

        fold_importance_df = pd.DataFrame()

        fold_importance_df['feature'] = train.columns.values

        fold_importance_df['importance'] = importances

        fold_importance_df['fold'] = i

        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)        

        i += 1

    print('{} cv RMSE scores : {}'.format(label, cv_scores))

    print('{} cv mean RMSE score : {}'.format(label, np.mean(cv_scores)))

    print('{} cv std RMSE score : {}'.format(label, np.mean(cv_scores)))

    print('{} cv QWK scores : {}'.format(label, qwk_scores))

    print('{} cv mean QWK score : {}'.format(label, np.mean(qwk_scores)))

    print('{} cv std QWK score : {}'.format(label, np.std(qwk_scores)))

    pred_full_test = pred_full_test / N_SPLITS

    results = {'label': label,

               'train': pred_train, 'test': pred_full_test,

                'cv': cv_scores, 'qwk': qwk_scores,

               'importance': feature_importance_df,

               'coefficients': all_coefficients}

    return results



params = {

          'depth': 8,

          'eta': 0.03,

          'task_type' :"GPU",

          'random_strength': 1.5,

          'loss_function': 'RMSE',

#           'one_hot_max_size': 2,

          'reg_lambda': 6,

          'od_type': 'Iter',

#           'fold_len_multiplier': 2,

          'border_count': 128,

#           # 'od_type': 'IncToDec',

#           # 'od_pval': 10e-5, 



    

          'bootstrap_type' : "Bayesian",

#           'bagging_temperature': 1,

          'random_seed': 123455,

          'verbose_eval': 100,

          'early_stopping_rounds': 100, 

          'num_boost_round': 2500}



def runLGB(train_X, train_y, test_X, test_y, test_X2, params):

    print('Prep LGB')

    watchlist = (test_X, test_y)

    print('Train LGB')

#     num_rounds = params.pop('num_rounds')

    verbose_eval = params.pop('verbose_eval')

    early_stop = None

    if params.get('early_stop'):

        early_stop = params.pop('early_stop')



    model = CatBoostRegressor(cat_features=list(cat_features), **params)



    model.fit(train_X, train_y, eval_set=watchlist, verbose=verbose_eval)

    print('Predict 1/2')

    

    pred_test_y = model.predict(test_X)

    

    if len(pred_test_y.shape) == 1:

        optR = OptimizedRounder()

        optR.fit(pred_test_y, test_y.values)

        coefficients = optR.coefficients()

        pred_test_y_k = optR.predict(pred_test_y, coefficients)

        print("Valid Counts = ", Counter(test_y))

        print("Predicted Counts = ", Counter(pred_test_y_k))

        print("Coefficients = ", coefficients)

    # handle case where we output probabilities instead of floats

    else:

        pred_test_y = pred_test_y_k = np.argmax(pred_test_y, axis=1)

        coefficients = np.array([0.5, 1.5, 2.5, 3.5])

        print("Valid Counts = ", Counter(test_y))

        print("Predicted Counts = ", Counter(pred_test_y_k))

        

    

    qwk = quadratic_weighted_kappa(test_y, pred_test_y_k)

    print("QWK = ", qwk)

    print('Predict 2/2')

    

    pred_test_y2 = model.predict(test_X2)

    

    # handle case where we output probabilities instead of floats

    if len(pred_test_y.shape) != 1:

        pred_test_y2 = np.argmax(pred_test_y2, axis=1)

        

    return pred_test_y.reshape(-1, 1), pred_test_y2.reshape(-1, 1), model.feature_importances_, coefficients, qwk



results = run_cv_model(train, test, target, runLGB, params, rmse, 'lgb')
imports = results['importance'].groupby('feature')['feature', 'importance'].mean().reset_index()

imports.sort_values('importance', ascending=False)
optR = OptimizedRounder()

coefficients_ = np.mean(results['coefficients'], axis=0)

# manually adjust coefs

coefficients_[0] = 1.64

coefficients_[1] = 2.15

coefficients_[3] = 2.85

print(coefficients_)

train_predictions = [r[0] for r in results['train']]

train_predictions = optR.predict(train_predictions, coefficients_).astype(int)

Counter(train_predictions)
print("Overall Train QWK:", quadratic_weighted_kappa(target, train_predictions))
tolerance = 3e-3

coefs = np.mean(results['coefficients'], axis=0)

test_predictions = [r[0] for r in results['test']]

test_preds_ = optR.predict(test_predictions, coefficients_).astype(int)



def create_distribution(predictions, coefs):

    # initialize an array for our test distribution in case we are missing some values

    test_dist = np.zeros((5))

    

    test_dist_temp = pd.value_counts(predictions, normalize=True).sort_index()

    

    # create our test distribution including zero values

    for index, val in test_dist_temp.iteritems():

        test_dist[index] = val

    

    return test_dist



true_dist = pd.value_counts(target, normalize=True).sort_index()

test_dist = create_distribution(test_preds_, coefs)

print("True:", true_dist)



for i, _ in enumerate(coefs):

    lr = 0.02

    print("Tuning:", i, "value:", coefs[i])    

    print("Coefs:", coefs)

    print("Test:", test_dist)

    delta = true_dist[i] - test_dist[i]

    print("Delta:", delta)

    

    while abs(delta) > tolerance:

#         print("Learning rate:", lr)

        # adjust the coefficient

        if delta < 0:

            old_coef = coefs[i]

            

            coefs[i] -= lr

            test_preds_ = optR.predict(test_predictions, coefs).astype(int)

            test_dist = create_distribution(test_preds_, coefs)

            delta = true_dist[i] - test_dist[i]

            if delta < 0:

                coefs[i] = old_coef

                test_preds_ = optR.predict(test_predictions, coefs).astype(int)

                test_dist = create_distribution(test_preds_, coefs)

                break

                

        else:

            old_coef = coefs[i]

            coefs[i] += lr

            test_preds_ = optR.predict(test_predictions, coefs).astype(int)

            test_dist = create_distribution(test_preds_, coefs)

            delta = true_dist[i] - test_dist[i]

            if delta < 0:

                coefs[i] = old_coef

                test_preds_ = optR.predict(test_predictions, coefs).astype(int)

                test_dist = create_distribution(test_preds_, coefs)

                break

        

        # decay the learning rate so we hopefully converge at some point

        lr *= 0.99

        print("New coef:", coefs[i], "new dist:", test_dist[i], "new delta:", delta)

        

    print()
# do some more tuning on the last coefficient because we need to consider both the partition above and below it

i = 3

delta_1 = true_dist[i] - test_dist[i]

delta_2 = true_dist[i+1] - test_dist[i+1]

lr = 0.02



avg_delta = (abs(delta_1) + abs(delta_2)) / 2

print("D1:", delta_1)

print("D2:", delta_2)

print(coefs)

print("diff:", delta_2 - delta_1)



while abs(delta_2 - delta_1) > 3e-2:

    if (delta_2 - delta_1) > 0:

        coefs[i] -= lr

    else:

        coefs[i] += lr

    test_preds_ = optR.predict(test_predictions, coefs).astype(int)

    test_dist = create_distribution(test_preds_, coefs)

    delta_1 = true_dist[i] - test_dist[i]

    delta_2 = true_dist[i+1] - test_dist[i+1]

    lr *= 0.75

    

    if lr < 1e-8:

        break

#     print("D1:", delta_1)

#     print("D2:", delta_2)

#     print("diff:", delta_2 - delta_1)

#     print(lr)

print(coefs)

# override auto tuned coefs and use hand-tuned to see how performance changes

# coefs = coefficients_

# coefs[0] = 1.64

# coefs[1] = 2.15

# coefs[3] = 2.85
optR = OptimizedRounder()

test_predictions = [r[0] for r in results['test']]

test_predictions = optR.predict(test_predictions, coefs).astype(int)

Counter(test_predictions)
train_predictions = [r[0] for r in results['train']]

train_predictions = optR.predict(train_predictions, coefs).astype(int)

Counter(train_predictions)
print("True Distribution:")

print(pd.value_counts(target, normalize=True).sort_index())

print("Train Predicted Distribution:")

print(pd.value_counts(train_predictions, normalize=True).sort_index())

print("Test Predicted Distribution:")

print(pd.value_counts(test_predictions, normalize=True).sort_index())
pd.DataFrame(sk_cmatrix(target, train_predictions), index=list(range(5)), columns=list(range(5)))
print("Overall Train QWK:", quadratic_weighted_kappa(target, train_predictions))

rmse(target, [r[0] for r in results['train']])

submission = pd.DataFrame({'PetID': test_id, 'AdoptionSpeed': test_predictions})

submission.head()
submission.to_csv('submission.csv', index=False)