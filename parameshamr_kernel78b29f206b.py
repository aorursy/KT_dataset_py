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
train_toxic = pd.read_csv("../input/jigsaw-multilingual-toxic-comment-classification/jigsaw-toxic-comment-train.csv")
train_unintended = pd.read_csv("../input/jigsaw-multilingual-toxic-comment-classification/jigsaw-unintended-bias-train.csv")
x_col = 'comment_text'
y_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer()

X = vectorizer.fit_transform(train_toxic[x_col])
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import SGDClassifier
sc = []
for i in range(len(y_cols)):
    y = train_toxic[y_cols[i]]
    model = SGDClassifier(loss="hinge", penalty="l2", max_iter=5)
    scores = cross_val_score(model, X, y, cv=3)
    sc.append(scores.mean()*100)
    print(str(y_cols[i]) + " : " + str(scores.mean()*100))
    print(scores)
sum(sc) / len(sc)

