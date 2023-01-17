# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import cv2 as cv



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

# Any results you write to the current directory are saved as output.



#ignore warnings

import warnings

warnings.filterwarnings('ignore')

from sklearn.exceptions import ConvergenceWarning

warnings.simplefilter(action="ignore", category=ConvergenceWarning)
import math

import matplotlib as mpl

import matplotlib.pyplot as plt

import matplotlib.pylab as pylab

import seaborn as sns



#Common Model Algorithms

from sklearn import svm, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, gaussian_process

from xgboost import XGBClassifier

from catboost import CatBoostClassifier



#Common Model Helpers

from sklearn.preprocessing import OneHotEncoder, LabelEncoder,OrdinalEncoder, StandardScaler,KBinsDiscretizer

from sklearn import feature_selection

from sklearn import model_selection

from sklearn import metrics

from sklearn.pipeline import Pipeline

from sklearn.compose import ColumnTransformer

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.decomposition import TruncatedSVD



# Evaluation

from sklearn.metrics import cohen_kappa_score,make_scorer

from sklearn.model_selection import StratifiedKFold

from sklearn.model_selection import GridSearchCV
import os

import sys

import matplotlib.pyplot as plt

import matplotlib.pylab as pylab



import requests

from io import BytesIO

from PIL import Image

import numpy as np
!git clone https://github.com/Gurpreetsingh9465/imageCaption.git
os.chdir('imageCaption')
src = '/kaggle/input/image-captioning-pretrained-weights/optimized.pb'

dst = '/kaggle/working/imageCaption/optimized.pb'



# This creates a symbolic link on python in tmp directory

os.symlink(src, dst)
from model import ShowAndTellModel

from PIL import Image

import numpy as np

import argparse

import sys

import os





with open("dictionary.txt") as f:

    lines = f.read().split("\n")

    

word2token = {}

token2word = {}

for line in lines[:-1]:

    l = line.split('    ')

    word = l[0]

    token = int(l[1])

    word2token[word] = token

    token2word[token] = word

    

    

model = ShowAndTellModel('optimized.pb')

start_token="<S>"

end_token="</S>"

def getCaption(image):

    state = model.feed_image(image)

    cur_token = word2token[start_token]

    end = word2token[end_token]

    answere = ""

    for i in range(20):

        if cur_token == end:

            break

        t = np.array([cur_token])

        softmax,state,_ = model.inference_step(t,state)

        cur_token = np.argmax(softmax)

        if cur_token == word2token[end_token]:

            break

        answere += token2word[cur_token]+" "

    return answere
DATA_DIR = '/kaggle/input/petfinder-adoption-prediction/'



# Directory of images to run detection on

IMAGE_TRAIN_DIR = os.path.join(DATA_DIR, "train_images")
def extract(images_dir):

    captions=pd.DataFrame()

    failed_images = []

    counter = 0

    for filename in os.listdir(images_dir):

        try:

            image_number = filename[filename.find('-')+1 :-4]

            if(image_number not in ['1', '2', '3', '4']):

                continue   

            counter = counter +1

            if counter % 1000 == 0:

                print(counter)



            image = Image.open(os.path.join(images_dir, filename))

            image = np.array(image)

            caption = getCaption(image) 

            new_row = pd.Series({"imageId": filename , "caption": caption})

            captions = captions.append(new_row, ignore_index=True)



        except Exception as e:

            print(e)

            failed_images.append(filename)

            continue

    return captions
captions = extract(IMAGE_TRAIN_DIR)
captions.sample(10)
captions.to_csv('../captions_train.csv',index=False)
def add_image_captions(df, PetID):

    #df = caption df with imageId

    matching_rows = df[df['imageId'].str.contains(PetID)]

    count_row = matching_rows.shape[0]

        

    if count_row == 0:

        # no images for this pet

        return pd.Series()

    

    matching_rows['imageIndex'] = matching_rows.apply(lambda x: x['imageId'][x['imageId'].find('-')+1 :-4], axis=1)

    matching_rows = matching_rows.sort_values('imageIndex')

        

    # add values from all images

    pet_photo_values = pd.Series()

    for i, row in matching_rows.iterrows():

        img_nr = row['imageIndex']

        pet_photo_values = pet_photo_values.append(pd.Series(

            [row['caption']],                       

             index = ['caption_'+img_nr] ))

        

    return pet_photo_values
train_df = pd.read_csv("/kaggle/input/petfinder-adoption-prediction/train/train.csv")
result_df=pd.DataFrame()

result_df['PetID']=train_df['PetID']

result_df['AdoptionSpeed']=train_df['AdoptionSpeed']

result_df = result_df.merge(result_df.PetID.apply(lambda x: add_image_captions(captions, x)), left_index=True, right_index=True)

result_df.fillna("", inplace=True)
result_df.columns
result_df.to_csv('../train_captions.csv', index=False)
from sklearn.feature_extraction.text import TfidfVectorizer

# list of text documents

text =  result_df.caption_1 + result_df.caption_2 + result_df.caption_3 + result_df.caption_4

# create the transform

count_vec = TfidfVectorizer()

# tokenize and build vocab

count_vec.fit(text)

# summarize

print(count_vec.vocabulary_)

print(count_vec.idf_)
result_df_vec = pd.concat(

    [

        result_df,

        pd.DataFrame(count_vec.transform(result_df.caption_1).toarray()),

        pd.DataFrame(count_vec.transform(result_df.caption_2).toarray()),

        pd.DataFrame(count_vec.transform(result_df.caption_3).toarray()),

        pd.DataFrame(count_vec.transform(result_df.caption_4).toarray())

    ], axis=1

)
result_df_vec = result_df_vec.drop(['caption_1','caption_2','caption_3','caption_4' ], axis=1)
result_df_vec.columns = [str(x) for x in range(1,result_df_vec.shape[1]+1)]

result_df_vec.rename(columns={ result_df_vec.columns[0]: "PetID", result_df_vec.columns[1]: "AdoptionSpeed" }, inplace = True)
result_df_vec.sample(10)
#Machine Learning Algorithm (MLA) Selection and Initialization

MLA = [

    #Ensemble Methods

    ensemble.AdaBoostClassifier(),

    ensemble.BaggingClassifier(),

    ensemble.ExtraTreesClassifier(),

    ensemble.GradientBoostingClassifier(),

    ensemble.RandomForestClassifier(),

    

    #GLM

    linear_model.LogisticRegressionCV(),

    linear_model.PassiveAggressiveClassifier(),

    linear_model.RidgeClassifierCV(),

    linear_model.SGDClassifier(),

    linear_model.Perceptron(),

    

    #Navies Bayes

    naive_bayes.BernoulliNB(),

    naive_bayes.GaussianNB(),

    

    #Nearest Neighbor

    neighbors.KNeighborsClassifier(),

    

    #SVM

    svm.LinearSVC(),

    

    #Trees    

    tree.DecisionTreeClassifier(),

    tree.ExtraTreeClassifier(),

    

    #Discriminant Analysis

    discriminant_analysis.LinearDiscriminantAnalysis(),

    discriminant_analysis.QuadraticDiscriminantAnalysis(),



    #xgboost: 

    XGBClassifier(),

    

    #CatBoostClassifier(verbose=0)

    ]

def train_model(data, MLA_list = MLA):

    

    target = data['AdoptionSpeed']

    X_train = data.drop(['AdoptionSpeed'],axis=1)

    

    MLA_columns = ['MLA Name', 'MLA Parameters','MLA cohen_kappa_score','MLA Time']

    MLA_compare = pd.DataFrame(columns = MLA_columns)



    MLA_predict = data['AdoptionSpeed']

    

    row_index = 0

    for alg in MLA_list:



        MLA_name = alg.__class__.__name__

        MLA_compare.loc[row_index, 'MLA Name'] = MLA_name

        MLA_compare.loc[row_index, 'MLA Parameters'] = str(alg.get_params())

    

        kf = StratifiedKFold(n_splits=5, shuffle=True)

        kappa_score = make_scorer(cohen_kappa_score, weights='quadratic')

        cv_results = model_selection.cross_validate(alg, X_train, target, cv  = kf, scoring=kappa_score )

        

        MLA_compare.loc[row_index, 'MLA Time'] = cv_results['fit_time'].mean()

        MLA_compare.loc[row_index, 'MLA cohen_kappa_score'] = cv_results['test_score'].mean() 

        MLA_compare.loc[row_index, 'algo'] = alg.__class__

             

        #MLA_predict[MLA_name] = alg.predict(X_train)

        row_index+=1



    MLA_compare.sort_values(by = ['MLA cohen_kappa_score'], ascending = False, inplace = True)

    sns.barplot(x='MLA cohen_kappa_score', y = 'MLA Name', data = MLA_compare, color = 'b')

    plt.title('Machine Learning Algorithm Accuracy Score \n')

    plt.xlabel('Accuracy Score (%)')

    plt.ylabel('Algorithm')

    

    return MLA_compare
classifier_performance = train_model(result_df_vec.drop(['PetID'], axis=1))
X_test = result_df_vec.drop(['PetID', 'AdoptionSpeed'], axis=1)

target = result_df_vec['AdoptionSpeed']



best_classifier = classifier_performance.iloc[0][4]



classifier = best_classifier().fit(X_test, target)
# Directory of images to run detection on

TEST_IMAGE_DIR = '/kaggle/input/petfinder-adoption-prediction/test_images'

test_df = pd.read_csv("/kaggle/input/petfinder-adoption-prediction/test/test.csv")
captions_test = extract(TEST_IMAGE_DIR)
test_result_df=pd.DataFrame()

test_result_df['PetID']=test_df['PetID']

test_result_df = test_result_df.merge(test_result_df.PetID.apply(lambda x: add_image_captions(captions_test, x)), left_index=True, right_index=True)

test_result_df.fillna("", inplace=True)
test_result_df.to_csv('../test_captions.csv', index=False)
test_result_df_vec = pd.concat(

    [

        test_result_df,

        pd.DataFrame(count_vec.transform(test_result_df.caption_1).toarray()),

        pd.DataFrame(count_vec.transform(test_result_df.caption_2).toarray()),

        pd.DataFrame(count_vec.transform(test_result_df.caption_3).toarray()),

        pd.DataFrame(count_vec.transform(test_result_df.caption_4).toarray())

    ], axis=1

)
test_result_df_vec = test_result_df_vec.drop(['caption_1','caption_2','caption_3','caption_4' ], axis=1)
test_result_df_vec.columns = [str(x) for x in range(1,test_result_df_vec.shape[1]+1)]

test_result_df_vec.rename(columns={ test_result_df_vec.columns[0]: "PetID" }, inplace = True)
submit=pd.DataFrame()

submit['PetID']=test_result_df['PetID']

submit['AdoptionSpeed']=classifier.predict(test_result_df_vec.drop(['PetID'], axis=1))

submit['AdoptionSpeed']=submit['AdoptionSpeed'].astype(int)

submit.to_csv('../submission.csv',index=False)
X_train = result_df_vec.drop(['PetID'], axis=1)

X_test = test_result_df_vec.drop(['PetID'], axis=1)
X_train.columns = X_train.columns.astype(str)

X_test.columns = X_test.columns.astype(str)
print(X_train.shape)

print(X_test.shape)
xgb_params = {

    'eval_metric': 'rmse',

    'seed': 1337,

    'silent': 1,

}
import xgboost as xgb



def run_xgb(params, X_train, X_test):

    n_splits = 5

    verbose_eval = 1000

    num_rounds = 30000

    early_stop = 500



    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=1337)



    oof_train = np.zeros((X_train.shape[0]))

    oof_test = np.zeros((X_test.shape[0], n_splits))



    i = 0



    for train_idx, valid_idx in kf.split(X_train, X_train['AdoptionSpeed'].values):



        X_tr = X_train.iloc[train_idx, :]

        X_val = X_train.iloc[valid_idx, :]



        y_tr = X_tr['AdoptionSpeed'].values

        X_tr = X_tr.drop(['AdoptionSpeed'], axis=1)



        y_val = X_val['AdoptionSpeed'].values

        X_val = X_val.drop(['AdoptionSpeed'], axis=1)



        d_train = xgb.DMatrix(data=X_tr, label=y_tr, feature_names=X_tr.columns)

        d_valid = xgb.DMatrix(data=X_val, label=y_val, feature_names=X_val.columns)



        watchlist = [(d_train, 'train'), (d_valid, 'valid')]

        model = xgb.train(dtrain=d_train, num_boost_round=num_rounds, evals=watchlist,

                         early_stopping_rounds=early_stop, verbose_eval=verbose_eval, params=params)



        valid_pred = model.predict(xgb.DMatrix(X_val, feature_names=X_val.columns), ntree_limit=model.best_ntree_limit)

        test_pred = model.predict(xgb.DMatrix(X_test, feature_names=X_test.columns), ntree_limit=model.best_ntree_limit)



        oof_train[valid_idx] = valid_pred

        oof_test[:, i] = test_pred



        i += 1

    return model, oof_train, oof_test
model, oof_train, oof_test = run_xgb(xgb_params, X_train, X_test)
# from https://www.kaggle.com/naveenasaithambi/optimizedrounder-improved



import scipy as sp



from collections import Counter

from functools import partial

from math import sqrt



from sklearn.metrics import cohen_kappa_score, mean_squared_error

from sklearn.metrics import confusion_matrix as sk_cmatrix





# FROM: https://www.kaggle.com/myltykritik/simple-lgbm-image-features



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

        preds = pd.cut(X, [-np.inf] + list(np.sort(coef)) + [np.inf], labels = [0, 1, 2, 3, 4])

        return -cohen_kappa_score(y, preds, weights='quadratic')

    

    def fit(self, X, y):

        loss_partial = partial(self._kappa_loss, X = X, y = y)

        initial_coef = [0.5, 1.5, 2.5, 3.5]

        self.coef_ = sp.optimize.minimize(loss_partial, initial_coef, method='nelder-mead')

    

    def predict(self, X, coef):

        preds = pd.cut(X, [-np.inf] + list(np.sort(coef)) + [np.inf], labels = [0, 1, 2, 3, 4])

        return preds

    

    def coefficients(self):

        return self.coef_['x']
optR = OptimizedRounder()

optR.fit(oof_train, X_train['AdoptionSpeed'].values)

coefficients = optR.coefficients()

print(coefficients)

valid_pred = optR.predict(oof_train, coefficients)

qwk = quadratic_weighted_kappa(X_train['AdoptionSpeed'].values, valid_pred)

print("QWK = ", qwk)
coefficients_ = coefficients.copy()

#coefficients_[0] = 1.65

train_predictions = optR.predict(oof_train, coefficients_).astype(np.int8)

test_predictions = optR.predict(oof_test.mean(axis=1), coefficients_).astype(np.int8)
submission = pd.DataFrame({'PetID': test_result_df_vec['PetID'].values, 'AdoptionSpeed': test_predictions})

submission.to_csv('../submission_opt.csv', index=False)
import shutil

os.chdir('..')

shutil.rmtree('imageCaption')