# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

from os.path import join as jp



# Any results you write to the current directory are saved as output.

input_dir = jp(os.pardir, 'input')
def read_csv(subfolder, filename):

    _csv = pd.read_csv(jp(input_dir, subfolder, '{}.csv'.format(filename)))

    return _csv



train = read_csv('train', 'train')

train.columns
train.head()
import json

from tqdm import tqdm

MISSING_VALUE = np.NaN



def read_sentiments(subfolder, pet_ids):

    sentiments = []

    with tqdm(total=len(pet_ids), desc='Reading sentiment') as pbar:

        for pet_id in pet_ids:

            result = {'magnitude': MISSING_VALUE, 'score': MISSING_VALUE}

            filepath = jp(input_dir, subfolder, '{}.json'.format(pet_id))

            if os.path.isfile(filepath):

                with open(filepath) as f:

                    data = json.load(f)

                    result['magnitude'] = data['documentSentiment']['magnitude']

                    result['score'] = data['documentSentiment']['score']

            sentiments.append(result)

            pbar.update()

    return sentiments
def encode_color(r, g, b):

    _max = 255 * 256 ** 2 + 255 * 256 + 255

    value = r * 256 ** 2 + g * 256 + b

    return value / _max



def read_colors_metadata(subfolder, pet_ids, num_colors=4, prioritize_by='score'):

    

    def get_color_value(color, channel):

        c = color['color']

        if channel in c.keys():

            return c[channel]

        else:

            return 0.0

    

    metadata = []

    with tqdm(total=len(pet_ids), desc='Reading colors metadata') as pbar:

        for pet_id in pet_ids:

            result = {}

            filepath = jp(input_dir, subfolder, '{}-1.json'.format(pet_id))

            if os.path.isfile(filepath):

                with open(filepath) as f:

                    data = json.load(f)

                    colors = data['imagePropertiesAnnotation']['dominantColors']['colors']

                    colors = sorted(colors, key=lambda x: x[prioritize_by], reverse=True)[:num_colors]

                    for i, color in enumerate(colors):

                        r = get_color_value(color, 'red')

                        g = get_color_value(color, 'green')

                        b = get_color_value(color, 'blue')

                        result['color_{}'.format(str(i))] = encode_color(r, g, b)

                        result['score_{}'.format(str(i))] = color['score']

                        result['pixel_fraction_{}'.format(str(i))] = color['pixelFraction']

            else:

                for i in range(num_colors):

                    result['color_{}'.format(str(i))] = MISSING_VALUE

                    result['score_{}'.format(str(i))] = MISSING_VALUE

                    result['pixel_fraction_{}'.format(str(i))] = MISSING_VALUE



            metadata.append(result)

            pbar.update()

    return metadata
def read_image_metadata(subfolder, pet_ids):

    metadata = []

    with tqdm(total=len(pet_ids), desc='Reading images metadata') as pbar:

        for pet_id in pet_ids:

            result = {}

            filepath = jp(input_dir, subfolder, '{}-1.json'.format(pet_id))

            if os.path.isfile(filepath):

                with open(filepath) as f:

                    data = json.load(f)

                    try:

                        annotations = data['labelAnnotations']

                        top_annotation = annotations[0]

                        score = top_annotation['score']

                        description = top_annotation['description']

                        result['label_description'] = description

                        result['label_score'] = score

                    except:

                        result['label_description'] = MISSING_VALUE

                        result['label_score'] = MISSING_VALUE

            else:

                result['label_description'] = MISSING_VALUE

                result['label_score'] = MISSING_VALUE



            metadata.append(result)

            pbar.update()

    return metadata
import langid



def is_hex(value):

    try:

        int(value, 16)

        return True

    except:

        return False



def is_empty(value):

    return value == ''



def is_numeric(value):

    return isinstance(value, int) or isinstance(value, float)

    

def has_no_name(pbar):

    def _has_no_name(value):

        pbar.update()

        if isinstance(value, float) and np.isnan(value):

            return MISSING_VALUE

        if is_numeric(value):

            return True

        is_no_name = 'no name' in value.lower()

        is_none = 'none' in value.lower()

        return is_no_name or is_none

    return _has_no_name



def detect_language(pbar):

    def _detect_language(description):

        pbar.update()

        if isinstance(description, float) and np.isnan(description):

            return MISSING_VALUE

        lang = langid.classify(description)[0]

        if lang in ['fi', 'lb', 'pl', 'nb', 'eo', 'et', 'pt', 'lt',

                    'no', 'de', 'tl', 'nl', 'da', 'ro', 'fr', 'it',

                    'hr', 'la', 'sw', 'es', 'mg', 'mt', 'sl', 'eu',

                    'sv', 'ca', 'cs', 'sk', 'xh', 'hu']:

            lang = 'en'

        if lang in ['af', 'ms', 'jv', 'ja', 'bs']:

            lang = 'id'

        return lang

    return _detect_language



def calc_description_length(description):

    if isinstance(description, float) and np.isnan(description):

        return MISSING_VALUE

    result = len(description)

    if result == 0:

        return MISSING_VALUE

    return result
from PIL import Image



def get_image_sizes(df, subfolder):

    sizes = []

    with tqdm(total=len(df), desc='Image sizes calculation') as pbar:

        for row in df.itertuples():

            _id = row.PetID

            _photos = int(row.PhotoAmt)

            x = 0.0

            y = 0.0

            count = 0

            for i in range(_photos):

                image_name = '{}-{}.jpg'.format(_id, str(i + 1))

                image_path = jp(input_dir, subfolder, image_name)

                if os.path.isfile(image_path):

                    size = Image.open(image_path).size

                    x += size[0]

                    y += size[1]

                    count += 1

            if count > 0:

                x = x / _photos

                y = y / _photos

            else:

                x = MISSING_VALUE

                y = MISSING_VALUE

            sizes.append({'PetID': _id, 'width': x, 'height': y})

            pbar.update()

        return sizes
from sklearn.decomposition import TruncatedSVD

from sklearn.feature_extraction.text import TfidfVectorizer



def generate_text_features(df):

    desc = df.Description.values



    tfv = TfidfVectorizer(

        min_df=3,

        max_features=10000,

        strip_accents='unicode',

        analyzer='word',

        token_pattern=r'\w{1,}',

        ngram_range=(1, 3),

        use_idf=True,

        smooth_idf=True,

        sublinear_tf=True,

        stop_words='english')



    # Fit TFIDF

    tfv.fit(list(desc))

    transformed = tfv.transform(desc)

    print("tfidf:", transformed.shape)



    svd = TruncatedSVD(n_components=120)

    svd.fit(transformed)

    transformed = svd.transform(transformed)

    print("svd:", transformed.shape)

    return transformed
def preprocess(df, sizes, sentiments, colors_metadata, image_metadata):

    preprocessed = df.copy()

    preprocessed['DescLength'] = preprocessed.Description.apply(calc_description_length)

    preprocessed['Description'].fillna('', inplace=True)

    

    with tqdm(total=len(df), desc='Replacing names') as pbar:

        preprocessed['Name'] = np.where(preprocessed['Name'].apply(has_no_name(pbar)), 0, 1)



    with tqdm(total=len(df), desc='Detecting languages') as pbar:

        preprocessed['Lang'] = preprocessed.Description.apply(detect_language(pbar))



    preprocessed = pd.concat([preprocessed, pd.DataFrame(sizes)[['width', 'height']]], sort=False, axis=1)

    preprocessed = pd.concat([preprocessed, pd.DataFrame.from_dict(sentiments)], sort=False, axis=1)

    preprocessed = pd.concat([preprocessed, pd.DataFrame(colors_metadata)], sort=False, axis=1)

    preprocessed = pd.concat([preprocessed, pd.DataFrame(image_metadata)], sort=False, axis=1)

    return preprocessed



sizes = get_image_sizes(train, 'train_images')

sentiments = read_sentiments('train_sentiment', train.PetID.values)

colors_metadata = read_colors_metadata('train_metadata', train.PetID, prioritize_by='pixelFraction')

image_metadata = read_image_metadata('train_metadata', train.PetID)

train = preprocess(train, sizes, sentiments, colors_metadata, image_metadata)



train_svd = generate_text_features(train)

train_svd_df = pd.DataFrame(train_svd, columns=['svd_{}'.format(i) for i in range(train_svd.shape[1])])

train = pd.concat([train, train_svd_df], axis=1, sort=False)
train.head()
categorical_columns = [

    "Type", # 1 = Dog, 2 = Cat

    "Name", # Has name or not

    "Breed1", # Primary breed of pet (Refer to BreedLabels dictionary)

    "Breed2", # Secondary, if mixed

    "Gender", # 1 = Male, 2 = Female, 3 = Mixed, if profile represents group of pets

    "Color1", # Color 1 of pet (Refer to ColorLabels dictionary)

    "Color2",

    "Color3", 

    "MaturitySize", # Size at maturity (1 = Small, 2 = Medium, 3 = Large, 4 = Extra Large, 0 = Not Specified)

    "FurLength", # Fur length (1 = Short, 2 = Medium, 3 = Long, 0 = Not Specified)

    "Vaccinated", # Pet has been vaccinated (1 = Yes, 2 = No, 3 = Not Sure)

    "Dewormed", # Pet has been dewormed (1 = Yes, 2 = No, 3 = Not Sure)

    "Sterilized", # Pet has been spayed / neutered (1 = Yes, 2 = No, 3 = Not Sure)

    "Health", # Health Condition (1 = Healthy, 2 = Minor Injury, 3 = Serious Injury, 0 = Not Specified)

    "State", # State location in Malaysia (Refer to StateLabels dictionary)

    "Lang", # Language of the Description

    "label_description" # Description from image label annotation

]



categorical_indices = []

for cc in categorical_columns:

    for i, column in enumerate(train.columns):

        if column == cc:

            categorical_indices.append(i)

            break



categorical_indices
# Set type for categorical columns

def set_categorical_type(df):

    for cc in categorical_columns:

        df[cc] = df[cc].astype('category')

    return df



train = set_categorical_type(train)
def fill_missing_categories(df):

    preprocessed = df.copy()

    for cc in ['Color1', 'Color2', 'Color3', 'MaturitySize', 'FurLength', 'Health']:

        preprocessed[cc] = preprocessed[cc].apply(lambda x: MISSING_VALUE if x == 0 else x)

    return preprocessed



train = fill_missing_categories(train)
import lightgbm as lgb



def train_lgb(X_train, y_train, X_valid, y_valid, categorical_features=[]):

    lgb_train = lgb.Dataset(X_train, y_train, categorical_feature=categorical_features)

    lgb_valid = lgb.Dataset(X_valid, y_valid, reference=lgb_train, categorical_feature=categorical_features)

    params = {

        'application': 'regression',

        'boosting': 'gbdt',

        'metric': 'rmse',

        'num_leaves': 70,

        'max_depth': 8,

        'learning_rate': 0.002,

        'bagging_fraction': 0.85,

        'feature_fraction': 0.8,

        'min_split_gain': 0.02,

        'min_child_samples': 150,

        'min_child_weight': 0.02,

        'lambda_l2': 0.05,

        'verbosity': -1,

        'data_random_seed': 17,

        'early_stop': 100,

        'verbose_eval': 100,

        'num_rounds': 10000

    }



    gbm = lgb.train(params, lgb_train, num_boost_round=10000, valid_sets=lgb_valid, early_stopping_rounds=100)

    return gbm
irrelevant_columns = ['AdoptionSpeed', 'PetID', 'RescuerID', 'Description']

relevant_columns = [c for c in train.columns if c not in irrelevant_columns]

relevant_columns[:5]
from functools import partial

import scipy as sp



def quadratic_weighted_kappa(y, y_pred):

    """

    Calculates the quadratic weighted kappa

    axquadratic_weighted_kappa calculates the quadratic weighted kappa

    value, which is a measure of inter-rater agreement between two raters

    that provide discrete numeric ratings.  Potential values range from -1

    (representing complete disagreement) to 1 (representing complete

    agreement). A kappa value of 0 is expected if all agreement is due to

    chance.

    quadratic_weighted_kappa(rater_a, rater_b), where rater_a and rater_b

    each correspond to a list of integer ratings. These lists must have the

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
def predict_lgb(gbm, X_test):

    y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)

    return y_pred



def predict_xgb(xgb, X_test):

    dtest = xgboost.DMatrix(X_test)

    y_pred = xgb.predict(dtest, ntree_limit=xgb.best_ntree_limit)

    return y_pred
from sklearn.model_selection import StratifiedKFold



predictors = {}

valid_predictions = []



y = list(train['AdoptionSpeed'])

X = train['PetID']

skf = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)

for i, tpl in enumerate(skf.split(X, y)):

    train_index, valid_index = tpl

    train_df = train.iloc[train_index]

    valid_df = train.iloc[valid_index]

    

    X_train = train_df[relevant_columns]

    y_train = train_df[['AdoptionSpeed']]



    X_valid = valid_df[relevant_columns]

    y_valid = valid_df[['AdoptionSpeed']]

    

    gbm = train_lgb(X_train, y_train, X_valid, y_valid, categorical_features=categorical_indices)

    predictors['gbm_{}'.format(str(i))] = gbm

    valid_predictions.append(predict_lgb(gbm, X_valid))
X_test = read_csv('test', 'test')

test_sizes = get_image_sizes(X_test, 'test_images')

test_sentiments = read_sentiments('test_sentiment', X_test.PetID.values)

colors_metadata = read_colors_metadata('test_metadata', X_test.PetID, prioritize_by='pixelFraction')

image_metadata = read_image_metadata('test_metadata', X_test.PetID)



X_test = preprocess(X_test, test_sizes, test_sentiments, colors_metadata, image_metadata)

X_test = set_categorical_type(X_test)

X_test = fill_missing_categories(X_test)



test_svd = generate_text_features(X_test)

test_svd_df = pd.DataFrame(test_svd, columns=['svd_{}'.format(i) for i in range(test_svd.shape[1])])

X_test = pd.concat([X_test, test_svd_df], axis=1, sort=False)



X_test = X_test[relevant_columns]
X_test.head(10)
predictions = {}

for name, predictor in predictors.items():

    if 'gbm' in name:

        lgb_pred = predict_lgb(predictor, X_test)

        predictions[name] = lgb_pred

    elif 'xgb' in name:

        xgb_pred = predict_xgb(predictor, X_test)

        predictions[name] = xgb_pred
lgb_preds = np.zeros(predictions['gbm_0'].shape)

# xgb_preds = np.zeros(predictions['xgb_0'].shape)

for name, prediction in predictions.items():

    if 'gbm' in name:

        lgb_preds = lgb_preds + prediction

    elif 'xgb' in name:

        xgb_preds = xgb_preds + prediction
avg_final_preds = lgb_preds / 5

avg_final_preds[:10]
train_predictions = np.array([item for sublist in valid_predictions for item in sublist])

optimized_rounder = OptimizedRounder()

optimized_rounder.fit(train_predictions, train.AdoptionSpeed.values)

coefficients = optimized_rounder.coefficients()

rounded_final_preds = optimized_rounder.predict(avg_final_preds, coefficients).astype(np.int)

rounded_final_preds
submission = read_csv('test', 'sample_submission')

submission['AdoptionSpeed'] = rounded_final_preds

submission.head()
submission.to_csv('submission.csv', index=False)
from sklearn.metrics import mean_squared_error



y = list(train['AdoptionSpeed'])

X = train['PetID']

skf = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)

errors = []

kappas = []

for i, tpl in enumerate(skf.split(X, y)):

    _, valid_index = tpl

    valid_df = train.iloc[valid_index]

    y_pred = valid_predictions[i]

    y_valid = valid_df[['AdoptionSpeed']]

    rmse = mean_squared_error(y_valid.values, y_pred) ** 0.5

    errors.append(rmse)

    print('RMSE of prediction for {} fold is:'.format(str(i)), rmse)

    y_pred_int = optimized_rounder.predict(y_pred, coefficients).astype(np.int)

    kappa = quadratic_weighted_kappa(y_valid.values.T[0], y_pred_int)

    kappas.append(kappa)

    print('QWK of prediction for {} fold is:'.format(str(i)), kappa)

print('Mean RMSE', np.array(errors).mean())

print('Mean QWK', np.array(kappas).mean())
importances = {}

for j, tpl in enumerate(predictors.items()):

    name, predictor = tpl

    fis = predictor.feature_importance()

    fold_importances = []

    for i, fi in enumerate(fis):

        name = X_train.columns[i]

        fi_dict = {}

        fi_dict['Name'] = name

        fi_dict['Importance'] = fi

        fold_importances.append(fi_dict)

    importances[j] = fold_importances



pd.DataFrame(importances[2])

        

#     imports = results['importance'].groupby('feature')['feature', 'importance'].mean().reset_index()

#     imports.sort_values('importance', ascending=False)
import xgboost



def train_xgb(X_train, y_train, X_valid, y_valid):

    params = {

        'objective': 'reg:linear',

        'eval_metric': 'rmse',

        'eta': 0.001,

        'max_depth': 10,

        'subsample': 0.6,

        'colsample_bytree': 0.6,

        'alpha':0.001,

        'random_state': 42,

        'silent': False

    }



    tr_data = xgboost.DMatrix(X_train, y_train)

    va_data = xgboost.DMatrix(X_valid, y_valid)



    watchlist = [(tr_data, 'train'), (va_data, 'valid')]



    xgb = xgboost.train(params, tr_data, 2000, watchlist, maximize=False, early_stopping_rounds = 100, verbose_eval=100)

    return xgb
# for i, tpl in enumerate(skf.split(X, y)):

#     train_index, valid_index = tpl

#     train_df = train.iloc[train_index]

#     valid_df = train.iloc[valid_index]

    

#     X_train = train_df[relevant_columns]

#     y_train = train_df[['AdoptionSpeed']]



#     X_valid = valid_df[relevant_columns]

#     y_valid = valid_df[['AdoptionSpeed']]

    

#     xgb = train_xgb(X_train, y_train, X_valid, y_valid)

#     predictors['xgb_{}'.format(str(i))] = xgb