import scipy

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error

from math import sqrt

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

import nltk

nltk.download('stopwords')

from nltk.corpus import stopwords

stopWords = stopwords.words('russian')
df = pd.read_csv("../input/train_fe_3.csv",parse_dates=["activation_date"], index_col=0)

df['description'] = df['description'].fillna(' ')



count = CountVectorizer()

title = count.fit_transform(df['title'])



tfidf = TfidfVectorizer(max_features=50000, stop_words = stopWords)

description = tfidf.fit_transform(df['description'])

df_num = df.select_dtypes(exclude=['object','datetime64'])

df_num = df_num.drop(['price','width_height_ratio'],axis=1)

target = df_num.deal_probability

X = df_num.drop(['deal_probability'],axis = 1)



import gc 

del df

del df_num

del count

del tfidf

del train_downloaded

del nltk

del stopWords

gc.collect()
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X = scaler.fit_transform(X)  # Don't cheat - fit only on training data



X = scipy.sparse.hstack((X,

                         title,

                         description)).tocsr()

X_train, X_test, y_train, y_test = train_test_split(X, target, test_size = 0.40, random_state = 42)



del title

del X

del target

del scaler

gc.collect()
del X

del target

del scaler

gc.collect()
from sklearn.metrics import mean_absolute_error

from sklearn.metrics import explained_variance_score

from sklearn.metrics import r2_score

from sklearn.metrics import mean_squared_log_error

from sklearn.metrics import median_absolute_error
import time
df['activation_date'].head()
desc = df.describe(include=['O']).sort_values('unique', axis = 1)
def highlight_row(x):

    if x.name == 'unique':

        color = 'lightblue'

    else:

        color = ''

    return ['background-color: {}'.format(color) for val in x]
desc.style.apply(highlight_row, axis =1)
def color_zero_red(val):

    """

    Takes a scalar and returns a string with

    the css property `'background-color: red'` for negative

    strings, black otherwise.

    """

    color = 'red' if val == 0 else ' '

    return 'background-color: %s' % color
df.describe().style.applymap(color_zero_red)
(df.deal_probability ==0).sum()/df.shape[0]
df_num = df.select_dtypes(exclude=['object','datetime64'])

df_num = df_num.drop(['price'],axis=1)
df_num.info()
target = df_num.deal_probability

X = df_num.drop(['deal_probability'],axis = 1)
target.shape
X.info()
X.head()
X_train, X_test, y_train, y_test = train_test_split(X, target, test_size = 0.30, random_state = 42)
X_train.shape
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scaler.fit(X_train)  # Don't cheat - fit only on training data

X_train = scaler.transform(X_train)

X_test = scaler.transform(X_test)  # apply same transformation to test data
import lightgbm as lgb

import time
def run_lgb(train_X, train_y, val_X, val_y):

    params = {

        "objective" : "regression",

        "metric" : "rmse",

        "num_leaves" : 32,

        "learning_rate" : 0.02,

        "bagging_fraction" : 0.7,

        "feature_fraction" : 0.6,

        "bagging_frequency" : 5,

        "bagging_seed" : 42,

        "verbosity" : -1,

        "max_depth" : 15

    }

    t1=time.time()

    lgtrain = lgb.Dataset(train_X, label=train_y)

    lgval = lgb.Dataset(val_X, label=val_y)

    evals_result = {}

    model = lgb.train(params, lgtrain, 300, valid_sets=[lgval], verbose_eval=1000, evals_result=evals_result)

    print("training time:", round(time.time()-t1, 3), "s")

    preds = model.predict(X_test)

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

    return model
# Training the model #

# v1 best RMSE:  0.2278794731156375

t0 = time.time()

model = run_lgb(X_train, y_train, X_test, y_test)

t1 = time.time()

print('training time: ' + str(t1-t0))