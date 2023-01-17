import numpy as np

import pandas as pd

import tensorflow as tf

import matplotlib.pyplot as plt



from sklearn.preprocessing import LabelEncoder

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from sklearn.svm import SVR

from sklearn.linear_model import ElasticNet

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV



from scipy.stats import uniform, randint
df_train = pd.read_csv('/kaggle/input/mercedes-benz-greener-manufacturing/train.csv.zip')

df_test = pd.read_csv('/kaggle/input/mercedes-benz-greener-manufacturing/test.csv.zip')

submission = pd.read_csv('/kaggle/input/mercedes-benz-greener-manufacturing/sample_submission.csv.zip')
df_train
df_train['y'].hist(bins=100)
threshold = df_train['y'].quantile(0.99)



where = df_train['y'] <= threshold



df_train = df_train[where]



df_train['y'].hist(bins=100)
std = df_train.std()



columns = std[std == 0].index



df_train = df_train.drop(columns=columns)



df_train
df_test = df_test.drop(columns=columns)



df_test
for column in df_train:

    

    if df_train[column].dtypes == 'object':

        

        X = [df_train[column], df_test[column]]

        X = np.hstack(X)

        

        enc = LabelEncoder()

        enc.fit(X)

        

        df_train[column] = enc.transform(df_train[column])

        df_test[column] = enc.transform(df_test[column])

        

df_train
X_train = df_train.loc[:, 'X0':]

y_train = df_train.loc[:, 'y']
param_random = {

    'learning_rate': uniform(0.01, 0.1),

    'n_estimators': randint(100, 300),

    'max_depth': randint(3, 5)

}

model_gbr = RandomizedSearchCV(GradientBoostingRegressor(), param_random, n_jobs=-1)

model_gbr.fit(X_train, y_train)

model_gbr.best_score_
param_random = {

    'C': uniform(0.1, 3.0),

}

model_svr = RandomizedSearchCV(SVR(), param_random, n_jobs=-1)

model_svr.fit(X_train, y_train)

model_svr.best_score_
param_random = {

    'alpha': uniform(0.01, 100),

    'l1_ratio': uniform(0.25, 0.75)

}

model_esn = RandomizedSearchCV(ElasticNet(), param_random, n_jobs=-1, n_iter=100)

model_esn.fit(X_train, y_train)

model_esn.best_score_
X_test = df_test.loc[:, 'X0':]

y_test = 0.5 * model_gbr.predict(X_test) + 0.1 * model_svr.predict(X_test) + 0.4 * model_esn.predict(X_test)

submission['y'] = y_test

submission.to_csv('submssion.csv', index=False)