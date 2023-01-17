# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
from sklearn.feature_extraction.text import TfidfVectorizer
df = pd.read_csv('../input/train.csv')
df.replace([np.NaN],[''],inplace=True)
X=df['question_text']
y=df['target']
df['question_text'] = df['question_text'].str.replace("[^a-zA-Z#]", " ")
train=df.question_text
test=df.target
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df['question_text'], df['target'], random_state = 0)

vect = TfidfVectorizer(min_df = 5).fit(X_train)
X_train_vectorized = vect.transform(X_train)
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train_vectorized, y_train)
predictions = model.predict(vect.transform(X_test))
from sklearn.metrics import roc_auc_score
print('AUC: ', roc_auc_score(y_test, predictions))
print(model.predict(vect.transform(['Has the United States become the largest dictatorship in the world'])))