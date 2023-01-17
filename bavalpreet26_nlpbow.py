# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import nltk

nltk.download('wordnet')



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('/kaggle/input/imdb-dataset-of-50k-movie-reviews/IMDB Dataset.csv')

df.head()
import numpy as np

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(df["review"], df["sentiment"], test_size=0.33)
X_train.head()
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_df=0.95, min_df=5)

X = vectorizer.fit_transform(X_train)

X_test= vectorizer.transform(X_test)

print(vectorizer.get_feature_names())
print(X.shape,X_test.shape,y_train.shape,y_test.shape)
# X = X.toarray()
# from sklearn.linear_model import LogisticRegression

# lr = LogisticRegression(C=2, dual=True,solver='liblinear',max_iter=5000)



from sklearn.naive_bayes import MultinomialNB

clf = MultinomialNB()

y_pred = clf.fit(X, y_train).predict(X_test)
# lr.fit(X,y_train)
# prediction = lr.predict(X_test)
# pred = prediction.tolist() 

# y_test = y_test.tolist()

from sklearn.metrics import accuracy_score

from sklearn.metrics import precision_score

from sklearn.metrics import recall_score

from sklearn.metrics import f1_score

# accuracy: (tp + tn) / (p + n)

accuracy_with_1gram = accuracy_score(y_test, y_pred)

print('Accuracy: %f' % accuracy_with_1gram)
# # precision tp / (tp + fp)

# precision = precision_score(y_test, pred,average="binary", pos_label="neg")

# print('Precision: %f' % precision)

# # # recall: tp / (tp + fn)

# # recall = recall_score(y_test, pred)

# # print('Recall: %f' % recall)

# # # f1: 2 tp / (2 tp + fp + fn)

# # f1 = f1_score(y_test, pred)

# # print('F1 score: %f' % f1)