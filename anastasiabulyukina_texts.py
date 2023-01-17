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
data = fetch_20newsgroups(subset="all",categories=["alt.atheism","sci.space"])
tmp = pd.DataFrame(data["data"])

tmp["target"] = data["target"]
tmp.head()
tmp.loc[2][0]
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()

X = vectorizer.fit_transform(data["data"])
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()

lr.fit(X,data["target"])
lr.score(X,data["target"])
lr.predict(X[:5])
lr.predict(vectorizer.transform(["going to fly my speceship",

                                "do you belive in god"]))