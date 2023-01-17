# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
column_sites = [ f"site{i}" for i in range(1,11)]
column_dates = [ f"time{i}" for i in range(1,11)]
KG_PATH='/kaggle/input/catch-me-if-you-can-intruder-detection-through-webpage-session-tracking2/'
df_train = pd.read_csv(KG_PATH+'train_sessions.csv',index_col='session_id',parse_dates=column_dates)
df_test  = pd.read_csv(KG_PATH+'test_sessions.csv' ,index_col='session_id',parse_dates=column_dates)
df_train.head(3)
df_train[column_sites] = df_train[column_sites].fillna(0).astype('str')
df_test [column_sites] = df_test [column_sites].fillna(0).astype('str')
from sklearn.feature_extraction.text import CountVectorizer

X_train = df_train[column_sites]
X_train = X_train.apply(lambda x: ' '.join(x),axis=1).values
X_test  = df_test [column_sites]
X_test  = X_test .apply(lambda x: ' '.join(x),axis=1).values
y_train = df_train.target
count_vectorizer = CountVectorizer (ngram_range=(1,5),max_features=50000)
X_train = count_vectorizer.fit_transform(X_train)
X_test  = count_vectorizer.transform    (X_test )
from sklearn.linear_model import LogisticRegression
logit = LogisticRegression(random_state=42,solver='liblinear')
logit.fit(X_train,y_train)
y_pred  = logit.predict_proba(X_test)[:,1]
df_pred = pd.DataFrame(y_pred,index=np.arange(1,y_pred.shape[0] + 1),columns=['target']) 
df_pred.to_csv(f'submission_alice_igorzenkov.csv',index_label='session_id')