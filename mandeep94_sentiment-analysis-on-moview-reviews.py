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
train = pd.read_csv("../input/train.tsv", sep="\t")
test = pd.read_csv("../input/test.tsv", sep="\t")
sampleSub = pd.read_csv("../input/sampleSubmission.csv")
train=train.set_index('PhraseId')
train.head()
train.isnull().any()
train['Sentiment'].value_counts()
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(train['Phrase'], train.Sentiment, random_state=0)
from sklearn.feature_extraction.text import CountVectorizer
vect = CountVectorizer(min_df=5, ngram_range=(1, 10)).fit(X_train)
X_train_vectorized = vect.transform(X_train)
len(vect.get_feature_names())
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train_vectorized, y_train)

predictions = model.predict(vect.transform(X_test))

from sklearn.metrics import f1_score
print('f1 score: ', f1_score(y_test, predictions, average="micro"))
feature_names = np.array(vect.get_feature_names())

sorted_coef_index = model.coef_[0].argsort()

print('Smallest Coefs:\n{}\n'.format(feature_names[sorted_coef_index[:10]]))
print('Largest Coefs: \n{}'.format(feature_names[sorted_coef_index[:-11:-1]]))
test = test.set_index('PhraseId')
result = model.predict(vect.transform(test.Phrase))
result = pd.Series(result)
test = test.reset_index()
test.head()
result
test['Sentiment'] = result
test.head()
test[['PhraseId', 'Sentiment']].to_csv('submission.csv',encoding="utf-8", index=False)