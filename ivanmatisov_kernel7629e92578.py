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

nextwesgroup = fetch_20newsgroups(subset="all" ,

                                  categories=["alt.atheism" ,"sci.space","comp.graphics" ])
nextwesgroup.keys()
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(nextwesgroup["data"])
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()
lr.fit(X,nextwesgroup["target"])
lr.score(X,nextwesgroup["target"])
nextwesgroup["data"][:3]
nextwesgroup["target"]
lr.predict(vectorizer.transform(["going to take my space ship shuttle","God, Atheism","Computers"]))