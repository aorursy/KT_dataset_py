# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt  #data visualisition

import seaborn as sns #data visualisition



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df_train = pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')

df_test = pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')

df_sub = pd.read_csv('/kaggle/input/nlp-getting-started/sample_submission.csv')
df_train.info()

df_test.info()
sns.countplot(x=df_train['target'], data=df_train)

plt.title("TARGET DISTRIBUTION", fontsize = 20)

plt.xlabel("Target Values", fontsize = 15)

plt.ylabel("Count", fontsize = 15)

plt.show()
y_train = df_train['target']

test_id = df_test['id']

df_train.drop(['target', 'id'], axis=1, inplace=True)

df_test.drop('id', axis=1, inplace=True)
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(min_df=5, max_df=0.5, ngram_range=(1, 2))

x_train = tfidf.fit_transform(df_train['text'].values)

x_test = tfidf.transform(df_test["text"].values)
import lightgbm as lgb

classifier = lgb.LGBMClassifier(**{

                    'learning_rate': 0.05,

                    'feature_fraction': 0.1,

                    'min_data_in_leaf' : 12,

                    'max_depth': 3,

                    'reg_alpha': 1,

                    'reg_lambda': 1,

                    'objective': 'binary',

                    'metric': 'auc',

                    'n_jobs': -1,

                    'n_estimators' : 5000,

                    'feature_fraction_seed': 42,

                    'bagging_seed': 42,

                    'boosting_type': 'gbdt',

                    'verbose': 1,

                    'is_unbalance': False,

                    'boost_from_average': False})


classifier.fit(x_train, y_train)
# Predicting the Test set results

y_pred = classifier.predict(x_test)
#Sumbmission the result

df_sub = pd.DataFrame()

df_sub['id'] = test_id

df_sub['target'] = y_pred

df_sub.to_csv('submission.csv', index=False)