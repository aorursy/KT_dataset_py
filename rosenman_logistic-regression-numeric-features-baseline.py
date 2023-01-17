import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.metrics import accuracy_score

# import module we'll need to import our custom module
from shutil import copyfile

# copy our file into the working directory (make sure it has .py suffix)
copyfile(src = "../input/lossespy/Losses.py", dst = "../working/Losses.py")

# import all our functions
from Losses import *
splitYear = 2014
from kaggle.competitions import twosigmanews
# You can only call make_env() once, so don't lose it!
env = twosigmanews.make_env()
print('Done!')
(market_train_df, news_train_df) = env.get_training_data()
market_train_df.shape, news_train_df.shape
market_train_df.head()
cat_cols = ['assetCode']
num_cols = ['volume', 'close', 'open', 'returnsClosePrevRaw1', 'returnsOpenPrevRaw1', 'returnsClosePrevMktres1',
                    'returnsOpenPrevMktres1', 'returnsClosePrevRaw10', 'returnsOpenPrevRaw10', 'returnsClosePrevMktres10',
                    'returnsOpenPrevMktres10']
from sklearn.preprocessing import StandardScaler
 
market_train_df[num_cols] = market_train_df[num_cols].fillna(0)
print('scaling numerical columns')

scaler = StandardScaler()

#col_mean = market_train[col].mean()
#market_train[col].fillna(col_mean, inplace=True)
scaler = StandardScaler()
market_train_df[num_cols] = scaler.fit_transform(market_train_df[num_cols])

#col_mean = np.mean(market_train_df.returnsOpenPrevMktres10)
#market_train_df['returnsOpenPrevMktres10'].fillna(col_mean, inplace=True)
market_train_df['y'] = ((market_train_df.returnsOpenNextMktres10 > 0).values).astype(int)
market_train_df['year'] = pd.to_datetime(market_train_df.time).dt.year
train = market_train_df[market_train_df.year <= splitYear]
test = market_train_df[market_train_df.year > splitYear]
def get_input(market_train, indices):
    X_num = market_train.loc[indices, num_cols].values
    X = {'num':X_num}
    for cat in cat_cols:
        X[cat] = market_train.loc[indices, cat_cols].values
    y = (market_train.loc[indices,'returnsOpenNextMktres10'] >= 0).values
    r = market_train.loc[indices,'returnsOpenNextMktres10'].values
    u = market_train.loc[indices, 'universe']
    d = market_train.loc[indices, 'time'].dt.date
    return X,y,r,u,d

# r, u and d are used to calculate the scoring metric
test_indices = np.where(market_train_df.year > splitYear)[0]
train_indices = np.where(market_train_df.year <= splitYear)[0]
X_train, y_train, r_train, u_train, d_train = get_input(market_train_df, train_indices)
X_test, y_test, r_test, u_test, d_test = get_input(market_train_df, test_indices)
colsToUse = ['volume',
'close',
'open',
'returnsClosePrevRaw1',
'returnsOpenPrevRaw1',
'returnsClosePrevMktres1',
'returnsOpenPrevMktres1',
'returnsClosePrevRaw10',
'returnsOpenPrevRaw10',
'returnsClosePrevMktres10',
'returnsOpenPrevMktres10']
# fit the models
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(C = 1)
X = train[colsToUse]
lr.fit(X = X, y = train.y)
# get the test and train predictions
Xtest = test[colsToUse]
probs_test = lr.predict_proba(Xtest)[:,1]
confidence_test = 2*(probs_test - 0.5)
plt.hist(confidence_test, bins='auto')

Xtrain = train[colsToUse]
probs_train = lr.predict_proba(Xtrain)[:,1]
confidence_train = 2*(probs_train - 0.5)
def computeSigmaScore(preds, r, u, d):
    x_t_i = preds * r * u
    data = {'day' : d, 'x_t_i' : x_t_i}
    df = pd.DataFrame(data)
    x_t = df.groupby('day').sum().values.flatten()
    mean = np.mean(x_t)
    std = np.std(x_t)
    score_valid = mean / std
    return(score_valid)
def computeCrossEntropyLoss(probs, r, eps = 1e-12):
    labels = (r >= 0).astype(int)
    probs_clipped = np.clip(probs, eps, 1.0-eps)
    return(np.mean(labels*np.log(probs_clipped) + (1-labels)*np.log(1-probs_clipped)))
[computeSigmaScore(confidence_test, r_test, u_test, d_test), 
 computeCrossEntropyLoss(probs_test, r_test)]
[computeSigmaScore(confidence_train, r_train, u_train, d_train), 
 computeCrossEntropyLoss(probs_train, r_train)]