import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



from scipy.sparse import csr_matrix, hstack



import time

import re

import math



from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler,LabelBinarizer

from sklearn.cross_validation import train_test_split

from sklearn.model_selection import GridSearchCV

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from sklearn.metrics import mean_squared_log_error

from sklearn.linear_model import Ridge

from sklearn.ensemble import GradientBoostingRegressor



import xgboost as xgb



seed = 90
Time_0 = time.time()

train = pd.read_csv('../input/train.tsv',sep='\t')

test = pd.read_csv('../input/test.tsv',sep='\t')
#Get log price

y_train = train['log_price'] = np.log((train['price'] + 1))
#Data prep functions

def handle_missing(dataset):

    dataset['category_name'].fillna(value="NA/NA/NA", inplace=True)

    dataset['brand_name'].fillna(value="missing", inplace=True)

    dataset['item_description'].fillna(value="missing", inplace=True)

    return (dataset)



def split_cat(dataset):

    dataset['cat1'], dataset['cat2'], dataset['cat3'] =  zip(*dataset['category_name'].str.split("/",2))

    return dataset



def label_maker(dataset):

    

    lb = LabelBinarizer(sparse_output=True)

    

    cat1 = lb.fit_transform(dataset['cat1'])

    cat2 = lb.fit_transform(dataset['cat2'])

    cat3 = lb.fit_transform(dataset['cat3'])

    brand_name = lb.fit_transform(dataset['brand_name'])

    

    del lb

    

    return cat1,cat2,cat3,brand_name



def get_dums(dataset):

    X_dummies = csr_matrix(pd.get_dummies(dataset[['item_condition_id', 'shipping']],

                                          sparse=True).values)

    

    return X_dummies



def text_processing(dataset):

    MIN_DF_COUNT = 10

    MAX_DF_COUNT = 10000

    cv = CountVectorizer(min_df = MIN_DF_COUNT, max_df = MAX_DF_COUNT)

    name = cv.fit_transform(dataset['name'])

    

    MIN_DF_TF = 10

    MAX_DF_TF = 51000

    MAX_FEATURES_TF = 51000

    

    tv = TfidfVectorizer(max_features=MAX_FEATURES_TF,

                         min_df = MIN_DF_TF,

                         max_df = MAX_DF_TF,

                         ngram_range=(1, 3),

                         stop_words='english')

    description = tv.fit_transform(dataset['item_description'])

    

    del cv, tv

    

    return name, description



#Merge dataset

nrow_train = train.shape[0]

merge: pd.DataFrame = pd.concat([train, test])

submission: pd.DataFrame = test[['test_id']]

    

del train

del test
#Preparing training data

# Time ~ 9 mins

start_time = time.time()



print("Handle Missing...")

merge = handle_missing(merge)



print("splitting cat...")

merge = split_cat(merge)



print("making labels...")

cat1,cat2,cat3,brand_name = label_maker(merge)



print("getting dummies...")

X_dummies = get_dums(merge)



print("processing text...")

name,description = text_processing(merge)



print("stacking train...")

sparse_merge = hstack((cat1,cat3,cat3,brand_name,X_dummies,name,description)).tocsr()



print("TIME:", time.time() - start_time)
#Split data

X_train = sparse_merge[:nrow_train]

X_test = sparse_merge[nrow_train:]
#Model building functions

def model_testing(model,X_test, y_test):

    y_pred = model.predict(X_test)

    error = rmsle(y_test, y_pred)

    print(error)

    



def rmsle(y, y0):

    assert len(y) == len(y0)

    return np.sqrt(np.mean(np.power(np.log1p(y)-np.log1p(y0), 2)))



#Initiating models

ridge_model_1 = Ridge(alpha=5.0, fit_intercept=True, normalize=False, copy_X=True, 

                    max_iter=None, tol=0.001, solver='auto', random_state=None)

ridge_model_2 = Ridge(alpha=5.0, fit_intercept=True, normalize=False, copy_X=True, 

                    max_iter=None, tol=0.001, solver='sag', random_state=None)

ridge_model_3 = Ridge(alpha=5.0, fit_intercept=True, normalize=False, copy_X=True, 

                    max_iter=None, tol=0.001, solver='lsqr', random_state=None)

gbrt = GradientBoostingRegressor(max_depth = 2, n_estimators = 5, 

                                 learning_rate = 0.9,subsample=0.9)
#Model execution

#Time ~ 6mins

start_time = time.time()



print("train test splitting...")

X_t, X_v, y_t, y_v = train_test_split(X_train, y_train,test_size = 0.20)



print("training model...")

print("1")

ridge_model_1.fit(X_train, y_train)

model_testing(ridge_model_1, X_test = X_v, y_test = y_v)

#Current best: .1233



print("training model...")

print("2")

#ridge_model_2.fit(X_t, y_t)

#model_testing(ridge_model_2, X_test = X_v, y_test = y_v)



print("training model...")

print("3")

#ridge_model_3.fit(X_t, y_t)

#model_testing(ridge_model_3, X_test = X_v, y_test = y_v)



print("TIME:", time.time() - start_time)
#Submission functions

def create_submission(model,test = X_test, submission=submission,path="./predictions.csv"):

    predictions = model.predict(test)

    predictions = pd.Series(np.exp(predictions) - 1)

    

    submission['price'] = predictions

    

    submission.to_csv(path, index=False)

    

    print(submission.describe())
#Generating submission

#Time ~ 15 secs

start_time = time.time()



create_submission(ridge_model_1)



print("TIME:", time.time() - start_time)

print("TOTAL TIME:", time.time() - Time_0)

#Total Time ~ 