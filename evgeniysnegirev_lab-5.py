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

news = fetch_20newsgroups(subset="all" ,categories =[ "alt.atheism", "sci.space" ])

data = news["data"]

target = news["target"]

from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer()

X = vectorizer.fit_transform(data)

from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()

lr.fit(X, target)
lr.predict(vectorizer.transform(["Atheism, god and so on", "spaship shuttles are "]))
# Статья про космос: Looking at the stars on the sky at night and following the movement of the sun at the day time, experiencing solar and moon eclipses, northern lights and meteoric showers, people from ancient times have begun to think about space. It conceals many secrets and mysteries, and, even nowadays, at the period of high developed technological progress, the scientists are still puzzled with a great number of unresolved questions. Studying cosmos we hope to find another civilizations or at least simple forms of life.

# Статья про религию: Meanwhile their opponents are sure to say that people themselves make their choices and decisions and are responsible for what happens to them. A huge number of people all over the world profess some religion Christianity, Islam, Buddhism, Judalsm or some other — and find in their religions answers to the most difficult questions.

a = "Looking at the stars on the sky at night and following the movement of the sun at the day time, experiencing solar and moon eclipses, northern lights and meteoric showers, people from ancient times have begun to think about space. It conceals many secrets and mysteries, and, even nowadays, at the period of high developed technological progress, the scientists are still puzzled with a great number of unresolved questions. Studying cosmos we hope to find another civilizations or at least simple forms of life."

b = "Meanwhile their opponents are sure to say that people themselves make their choices and decisions and are responsible for what happens to them. A huge number of people all over the world profess some religion Christianity, Islam, Buddhism, Judalsm or some other — and find in their religions answers to the most difficult questions."
lr.predict(vectorizer.transform([a, b]))