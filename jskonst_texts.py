# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
from sklearn.datasets import fetch_20newsgroups
data = fetch_20newsgroups(subset="all",categories=["alt.atheism","sci.space" ])
data["data"][0]
data["target"][0]
tmp = pd.DataFrame(data["data"])
tmp["target"] = data["target"]
tmp.head(10)
tmp.loc[2]
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_df=0.5)
X = vectorizer.fit_transform(data["data"])
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X,data["target"])
lr.score(X,data["target"])
lr.predict(X[:10])
lr.predict(vectorizer.transform(["going to fly my",
                                     "do you belive"]))
res = vectorizer.transform(["going to fly my",
                                     "fly with my"])
same = 0
print( res.shape[1])
el1 = res[0]
el2 = res[1]
one = set(el1.nonzero()[1])
print("---------------------")
two = set(el2.nonzero()[1])

print(one.intersection(two))
# for i in range(0, res.shape[1]):
#     el1 = res[0][i]
#     el2 = res[1][i]
#     print(el1)
#     if el1 == el2:
#         same +=1
# print(same)