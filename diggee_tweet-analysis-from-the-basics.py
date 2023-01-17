# importing libraries



import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import string

import re

from nltk.corpus import stopwords

from sklearn.linear_model import RidgeClassifier, LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import SVC

from xgboost import XGBClassifier

from lightgbm import LGBMClassifier

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from sklearn.model_selection import cross_val_score

from bayes_opt import BayesianOptimization

import warnings

warnings.filterwarnings('ignore')
#%% reading data



def get_data():

    full_train_data = pd.read_csv('/kaggle/input/nlp-getting-started/train.csv', index_col = 'id', usecols = ['id','text','target'])

    full_test_data = pd.read_csv('/kaggle/input/nlp-getting-started/test.csv', index_col = 'id', usecols = ['id','text'])

    # there are no null values in either the train set or the test set.

    print('Total null values in the train set - '+str(full_train_data.isnull().sum().sum()))

    print('Total null values in the test set - '+str(full_test_data.isnull().sum().sum()))

    return full_train_data, full_test_data
#%% preprocessing data



def clean_data(full_train_data, full_test_data):

    

    def tweet_clean(tweet):

        # convert every tweet to lower case

        tweet = ''.join([j.lower() for j in tweet])

        # remove website links

        tweet = re.sub('www.|https://|http://|.com|t.co/','',tweet)    

        # remove all punctuation 

        tweet = ''.join([j for j in tweet if j not in string.punctuation])    

        # remove all digits

        tweet = ''.join([j for j in tweet if j not in string.digits])    

        # remove stopwords

        tweet = ' '.join([j for j in tweet.split() if j not in stopwords.words('english')])    

        # remove non ASCII characters

        tweet = ''.join([j for j in tweet if ord(j) < 128])

        return tweet

    

    full_train_data['text'] = full_train_data['text'].apply(lambda x: tweet_clean(x))

    full_test_data['text'] = full_test_data['text'].apply(lambda x: tweet_clean(x))

    return full_train_data, full_test_data 
# model to be used for tweet analysis



def nlp_model(full_train_data, full_test_data):

    

    CV = CountVectorizer(analyzer = 'char_wb', ngram_range = (1,5))     # parameters of CV can be played around with

    X = CV.fit_transform(full_train_data['text'])

    y = full_train_data['target']

    X_valid = CV.transform(full_test_data['text'])

    

    tfidf = TfidfTransformer()

    X = tfidf.fit_transform(X)

    X_valid= tfidf.transform(X_valid)

    

    return X, y, X_valid
#%% data scaling

    

def scaled_data(X, X_valid):

    from sklearn.preprocessing import StandardScaler

    scaler_X = StandardScaler()

    X = scaler_X.fit_transform(X)

    X_valid = scaler_X.transform(X_valid)

    return X, X_valid, scaler_X  
#%% regressor functions

    

def regressor_fn_optimised(X, y, X_valid, choice):      

    

    if choice == 1:                    

        def regressor_fn(C):            

            regressor = LogisticRegression(C = C)      

            cval = cross_val_score(regressor, X, y, scoring = 'balanced_accuracy', cv = 5)

            return cval.mean()

        pbounds = {'C': (0.1, 10)}

        

    elif choice == 2:                    

        def regressor_fn(alpha):            

            regressor = RidgeClassifier(alpha = alpha)        

            cval = cross_val_score(regressor, X, y, scoring = 'balanced_accuracy', cv = 5)

            return cval.mean()

        pbounds = {'alpha': (1, 1.1)}

        

    elif choice == 3:    

                

        def regressor_fn(n_neighbors):     

            n_neighbors = int(n_neighbors)

            regressor = KNeighborsClassifier(n_neighbors = n_neighbors)        

            cval = cross_val_score(regressor, X, y, scoring = 'balanced_accuracy', cv = 5)

            return cval.mean()

        pbounds = {'n_neighbors': (2,10)}

        

    elif choice == 4:           

        def regressor_fn(n_estimators, max_depth):     

            max_depth, n_estimators = int(max_depth), int(n_estimators)

            regressor = RandomForestClassifier(n_estimators = n_estimators, max_depth = max_depth)        

            cval = cross_val_score(regressor, X, y, scoring = 'balanced_accuracy', cv = 5, n_jobs = -1)

            return cval.mean()

        pbounds = {'n_estimators': (10, 500), 'max_depth': (2,20)}

        

    elif choice == 5: 

        X, X_valid, scaler_X = scaled_data(X, X_valid)      

        def regressor_fn(C, gamma):            

            regressor = SVC(C = C, kernel = 'rbf', gamma = gamma)        

            cval = cross_val_score(regressor, X, y, scoring = 'balanced_accuracy', cv = 5)

            return cval.mean()

        pbounds = {'C': (0.1, 100), 'gamma': (0.01, 100)}

        

    elif choice == 6:

        def regressor_fn(learning_rate, max_depth, n_estimators):            

            max_depth, n_estimators = int(max_depth), int(n_estimators)

            regressor = LGBMClassifier(learning_rate = learning_rate, max_depth = max_depth, n_estimators = n_estimators)        

            cval = cross_val_score(regressor, X, y, scoring = 'balanced_accuracy', cv = 5)

            return cval.mean()

        pbounds = {'learning_rate': (0.01, 1), 'max_depth': (2,40), 'n_estimators': (10, 500)}        

        

    else:

        def regressor_fn(learning_rate, max_depth, n_estimators):            

            max_depth, n_estimators = int(max_depth), int(n_estimators)

            regressor = XGBClassifier(learning_rate = learning_rate, max_depth = max_depth, n_estimators = n_estimators)        

            cval = cross_val_score(regressor, X, y, scoring = 'balanced_accuracy', cv = 3)

            return cval.mean()

        pbounds = {'learning_rate': (0.01, 1), 'max_depth': (2,50), 'n_estimators': (10, 500)}

    

    optimizer = BayesianOptimization(regressor_fn, pbounds, verbose = 2)

    optimizer.probe(params = {'C':1}, lazy = True)

    optimizer.maximize(init_points = 5, n_iter = 10)    

    # change next line in accordance with choice of regressor made

    # y_valid_pred = RandomForestClassifier(max_depth = int(optimizer.max['params']['max_depth']), n_estimators = int(optimizer.max['params']['max_depth'])).fit(X, y).predict(X_valid)

    y_valid_pred = LogisticRegression(C = optimizer.max['params']['C']).fit(X, y).predict(X_valid)

    

    return y_valid_pred, optimizer.max
if __name__ == '__main__':

    full_train_data, full_test_data = get_data()

    full_train_data, full_test_data = clean_data(full_train_data, full_test_data)

    X, y, X_valid = nlp_model(full_train_data, full_test_data)

    # uncomment the following line if the model's hyper parameters have to be optimised. 

#     y_valid_pred, optimal_params = regressor_fn_optimised(X, y, X_valid, choice = 1)

    #comment the following line if regressor_fn_optimised is being run

    y_valid_pred = LogisticRegression().fit(X, y).predict(X_valid)

    df = pd.DataFrame({'Id':full_test_data.index, 'Target':y_valid_pred})

    df.to_csv('prediction.csv', index = False)