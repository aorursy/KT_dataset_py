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
        file_name = os.path.join(dirname, filename)
print(file_name)
# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import json
import gzip
import pandas as pd
import numpy as np
import random
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
def parse(path):
    with open(path) as jf:
        for l in jf:
            yield json.loads(l)
pos_ratings, neg_ratings = [], []
pos_texts, neg_texts = [], []

for review in parse(file_name):
    if 'reviewText' in review.keys():
        if review['reviewText'] != '':
            if review['overall'] < 3.0:
                neg_texts.append(review['reviewText'])
                neg_ratings.append('Neg')

            elif review['overall'] > 3.0:
                pos_texts.append(review['reviewText'])
                pos_ratings.append('Pos')
print(len(neg_ratings))
print(len(pos_ratings))
idx = list(range(len(pos_ratings)))
random.seed(42)
sample_idx = random.sample(idx, len(neg_ratings))
sample_pos_texts = [pos_texts[i] for i in sample_idx]
sample_pos_ratings = [pos_ratings[i] for i in sample_idx]
texts = []
ratings =[]
texts.extend(neg_texts)
texts.extend(sample_pos_texts)
ratings.extend(neg_ratings)
ratings.extend(sample_pos_ratings)
new_idx = list(range(len(texts)))
random.seed(42)
random.shuffle(new_idx)
texts = [texts[i] for i in new_idx]
ratings = [ratings[i] for i in new_idx]
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)
X.shape
X_train, X_test, y_train, y_test = train_test_split(X, ratings, test_size = 0.3, random_state=42)
reg = LogisticRegression(solver='newton-cg')
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)
print(accuracy_score(y_test, y_pred))
print(f1_score(y_test, y_pred, pos_label='Pos'))
print(confusion_matrix(y_test, y_pred))
submit = pd.DataFrame([],columns=['Label'])
for i in range(len(y_pred)):
    submit.loc[i, 'Label'] = y_pred[i]
submit.to_csv('submit.csv', mode='w', header = True)
