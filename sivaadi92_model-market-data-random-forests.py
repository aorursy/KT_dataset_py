# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
# Ctrl + Shift + P
import os
print(os.listdir("../input"))

toy = True
# Any results you write to the current directory are saved as output.
from kaggle.competitions import twosigmanews
env = twosigmanews.make_env()

(market_train_df, news_train_df) = env.get_training_data()

# train_my_model(market_train_df, news_train_df)
print(market_train_df.shape, news_train_df.shape)
print(market_train_df.columns)
print(news_train_df.columns)
# We will reduce the number of samples for memory reasons
toy = False
if toy:
    market_train_df = market_train_df.tail(30000)
    news_train_df = news_train_df.tail(30000)
else:
    market_train_df = market_train_df.tail(3_000_000)
    news_train_df = news_train_df.tail(6_000_000)


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import chain
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier



market_train_df['time'] = market_train_df['time'].dt.floor('1D')
market_train_df = market_train_df.reset_index(drop=True)

# market_train_df.head(5)

X_pruned = market_train_df[['returnsOpenPrevMktres10', 'returnsOpenPrevMktres1']]
Y_pruned = market_train_df[['returnsOpenNextMktres10']]

start_test_data = int(0.8 * market_train_df.shape[0])
X_train = X_pruned.iloc[:start_test_data]
Y_train = Y_pruned.iloc[:start_test_data]

X_test = X_pruned.iloc[start_test_data:].reset_index(drop=True)
Y_test = Y_pruned.iloc[start_test_data:].reset_index(drop=True)
Y_test_metadata = market_train_df.iloc[start_test_data:][['time', 'universe']].reset_index(drop=True)


Y_test_metadata
Y_train.hist(bins=10)
#Hyperparameter : threshold
def assignConfidence(pthreshold, nthreshold, Y):
    Y[(Y['returnsOpenNextMktres10'] > pthreshold)]  = 1.0
    Y[(Y['returnsOpenNextMktres10'] < -nthreshold)] = -1.0
    Y[(Y != 1.0) & (Y != -1.0)] = 0.0
#     Y.hist()
    return Y

def train_model(X, Y, **kwargs):
    randForest = RandomForestClassifier(**kwargs)
    X = X.ffill()
    randForest.fit(X, Y['returnsOpenNextMktres10'])
    return randForest

def overridden_predict(trained_model, X_test):
    X_test = X_test.ffill()
    y_pred = trained_model.predict(X_test)
    y_pred = pd.DataFrame({'confidenceValue':y_pred})
    return y_pred

def sigma_score(Y_pred, Y_test):
    score = Y_pred['confidenceValue'] * Y_test['returnsOpenNextMktres10'] * Y_test_metadata['universe']
    score = score.to_frame('score')
    score['time'] = Y_test_metadata['time'].values
    score_per_day = score.groupby('time').sum()
    try:
        sigma_score = score_per_day['score'].mean() / score_per_day['score'].std()
        print("The sigma score is *****" + str(sigma_score))
        if np.isnan(sigma_score):
            sigma_score = 0.0
    except:
        sigma_score = 0.0
        
    return sigma_score

#                 Y_train_modified = Y_train.copy()
#                 kwargs = {'max_depth':max_depth, 'n_estimators':number_of_bags}
#                 Y_train_modified = assignConfidence(pthresh, nthresh, Y_train_modified)
#                 trained_model = train_model(X_train, Y_train_modified, **kwargs)
#                 Y_pred = overridden_predict(trained_model, X_test)
#                 sig_score = sigma_score(Y_pred=Y_pred, Y_test=Y_test)
# Y_train_modified = Y_train.copy()
# max_depth = None
# number_of_bags = 20
# kwargs = {'max_depth':max_depth, 'n_estimators':number_of_bags}
# Y_train_modified = assignConfidence(0.05, 0.05, Y_train_modified)
# randomForest = RandomForestClassifier(**kwargs)
# X_train = X_train.ffill()
# # Y_train_modified.head(5)
# randomForest.fit(X_train, Y_train_modified['returnsOpenNextMktres10'])


# # sigma_score = score_per_day['score'].mean() / score_per_day['score'].std()

# Y_pred = overridden_predict(randomForest, X_test)
# # 
# score = Y_pred['confidenceValue'] * Y_test['returnsOpenNextMktres10'] * Y_test_metadata['universe']
# score = score.to_frame('score')
# score['time'] = Y_test_metadata['time'].values
# score_per_day = score.groupby('time').sum()
# score_per_day
# sig_score = sigma_score(Y_pred=Y_pred, Y_test=Y_test)
# sig_score
Y = pd.concat([Y_train, Y_test])
X = pd.concat([X_train, X_test])
best_pthresh = 0.075
best_nthresh = 0.1
best_max_depth = None
number_of_bags = 30

Y = pd.concat([Y_train, Y_test])
Y_modified = Y.copy()
Y_modified = assignConfidence(best_pthresh, best_nthresh, Y_modified)
kwargs = {'max_depth':best_max_depth, 'n_estimators':number_of_bags}
trained_model = train_model(X, Y_modified, **kwargs)
Y_pred = overridden_predict(trained_model, X)
def make_predictions(trained_model,predictions_template_df, market_obs_df, news_obs_df):
    sample = market_obs_df[['returnsOpenPrevMktres10', 'returnsOpenPrevMktres1']]
    sample = sample.ffill()
#     y_pred = lm.predict(sample)
    y_pred = trained_model.predict(sample)
    predictions_template_df.confidenceValue = y_pred.clip(-1, 1)

days = env.get_prediction_days()

for (market_obs_df, news_obs_df, predictions_template_df) in days:
    make_predictions(trained_model, predictions_template_df, market_obs_df, news_obs_df)
    env.predict(predictions_template_df)

print('Done!')


env.write_submission_file()
print("Fourth run")