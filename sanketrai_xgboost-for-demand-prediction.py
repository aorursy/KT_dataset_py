# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

import scipy

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error

from math import sqrt

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from nltk.corpus import stopwords

stopWords = stopwords.words('russian')

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/train_fe_3.csv",parse_dates=["activation_date"], index_col=0)
df['description'] = df['description'].fillna(' ')



count = CountVectorizer()

title = count.fit_transform(df['title'])



tfidf = TfidfVectorizer(max_features=50000, stop_words = stopWords)

description = tfidf.fit_transform(df['description'])
df.info()
df_num = df.select_dtypes(exclude=['object','datetime64'])

df_num = df_num.drop(['price','width_height_ratio'],axis=1)
df_num.info()
target = df_num.deal_probability

X = df_num.drop(['deal_probability'],axis = 1)
import gc 

del df

del df_num

gc.collect()
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X = scaler.fit_transform(X)  # Don't cheat - fit only on training data
X = scipy.sparse.hstack((X,

                         title,

                         description)).tocsr()
X_train, X_test, y_train, y_test = train_test_split(X, target, test_size = 0.40, random_state = 42)
del title

del tfidf

del count

del description

del X

del target

del scaler

gc.collect()
import xgboost as xgb

# Convert the training and testing sets into DMatrixes: DM_train, DM_test

DM_train = xgb.DMatrix(X_train, label = y_train)

DM_test = xgb.DMatrix(X_test, label = y_test)
del X_train

del X_test

del y_train

del y_test

gc.collect()
from sklearn.metrics import mean_absolute_error

from sklearn.metrics import explained_variance_score

from sklearn.metrics import r2_score

from sklearn.metrics import mean_squared_log_error

from sklearn.metrics import median_absolute_error
# Create the parameter dictionary: params

params={'seed':0,'colsample_bytree':0.8,'objective':'reg:linear','max_depth':24,'min_child_weight':24}



# Train the model: xg_reg

xg_reg = xgb.train(dtrain = DM_train, params=params)



# Predict the labels of the test set: preds

preds = xg_reg.predict(DM_test)
preds = np.clip(preds, 0, 1)



# Compute and print the result

mse = mean_squared_error(y_test,preds)

rmse = np.sqrt(mean_squared_error(y_test,preds))

mae = mean_absolute_error(y_test,preds)

exp = explained_variance_score(y_test,preds)

r2 = r2_score(y_test,preds,multioutput='variance_weighted')

mle = mean_squared_log_error(y_test,preds)

mdae = median_absolute_error(y_test,preds)



print("RMSE: %f" % (rmse))

print("MSE: %f" % (mse))

print("MAE: %f" % (mae))

print("EXP: %f" % (exp))

print("R2: %f" % (r2))

print("MLE: %f" % (mle))

print("MDAE: %f" % (mdae))

# Create the parameter dictionary: params

params={'seed':0,'colsample_bytree':0.8,'objective':'reg:logistic','max_depth':24,'min_child_weight':24}



# Train the model: xg_reg

xg_reg = xgb.train(dtrain = DM_train, params=params)



# Predict the labels of the test set: preds

preds = xg_reg.predict(DM_test)
preds = np.clip(preds, 0, 1)



# Compute and print the result

mse = mean_squared_error(y_test,preds)

rmse = np.sqrt(mean_squared_error(y_test,preds))

mae = mean_absolute_error(y_test,preds)

exp = explained_variance_score(y_test,preds)

r2 = r2_score(y_test,preds,multioutput='variance_weighted')

mle = mean_squared_log_error(y_test,preds)

mdae = median_absolute_error(y_test,preds)



print("RMSE: %f" % (rmse))

print("MSE: %f" % (mse))

print("MAE: %f" % (mae))

print("EXP: %f" % (exp))

print("R2: %f" % (r2))

print("MLE: %f" % (mle))

print("MDAE: %f" % (mdae))
# Create the parameter dictionary: params

params = {"objective":"reg:linear"}



# Train the model: xg_reg

xg_reg = xgb.train(dtrain = DM_train, params=params, num_boost_round= 300)



# Predict the labels of the test set: preds

preds = xg_reg.predict(DM_test)

preds = np.clip(preds, 0, 1)



# Compute and print the result

mse = mean_squared_error(y_test,preds)

rmse = np.sqrt(mean_squared_error(y_test,preds))

mae = mean_absolute_error(y_test,preds)

exp = explained_variance_score(y_test,preds)

r2 = r2_score(y_test,preds,multioutput='variance_weighted')

mle = mean_squared_log_error(y_test,preds)

mdae = median_absolute_error(y_test,preds)



print("RMSE: %f" % (rmse))

print("MSE: %f" % (mse))

print("MAE: %f" % (mae))

print("EXP: %f" % (exp))

print("R2: %f" % (r2))

print("MLE: %f" % (mle))

print("MDAE: %f" % (mdae))
import time

# Create the parameter dictionary: params

params = {"objective":"reg:logistic"}

t0=time.time()

# Train the model: xg_reg

xg_reg = xgb.train(dtrain = DM_train, params=params, num_boost_round= 300)

print("training time:", round(time.time()-t0, 3), "s")

# Predict the labels of the test set: preds

preds = xg_reg.predict(DM_test)



# Compute and print the result

preds = np.clip(preds, 0, 1)

mse = mean_squared_error(y_test,preds)

rmse = np.sqrt(mean_squared_error(y_test,preds))

mae = mean_absolute_error(y_test,preds)

exp = explained_variance_score(y_test,preds)

r2 = r2_score(y_test,preds,multioutput='variance_weighted')

mle = mean_squared_log_error(y_test,preds)

mdae = median_absolute_error(y_test,preds)



print("RMSE: %f" % (rmse))

print("MSE: %f" % (mse))

print("MAE: %f" % (mae))

print("EXP: %f" % (exp))

print("R2: %f" % (r2))

print("MLE: %f" % (mle))

print("MDAE: %f" % (mdae))
