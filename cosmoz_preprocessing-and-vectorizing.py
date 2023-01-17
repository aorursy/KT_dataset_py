# Imports

import pandas as pd

from pandas import Series,DataFrame



import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline



# machine learning

from sklearn.feature_extraction.text import CountVectorizer



# preprocessing

from bs4 import BeautifulSoup

from nltk.corpus import stopwords

import re

import string



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
dataframes = {

    "cooking": pd.read_csv("../input/cooking.csv"),

    "crypto": pd.read_csv("../input/crypto.csv"),

    "robotics": pd.read_csv("../input/robotics.csv"),

    "biology": pd.read_csv("../input/biology.csv"),

    "travel": pd.read_csv("../input/travel.csv"),

    "diy": pd.read_csv("../input/diy.csv"),

}



test = pd.read_csv("../input/test.csv")
# remove html tags and uris from contents



uri_re = r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))'



def stripTagsAndUris(x):

    if x:

        # BeautifulSoup on content

        soup = BeautifulSoup(x, "html.parser")

        # Stripping all <code> tags with their content if any

        if soup.code:

            soup.code.decompose()

        # Get all the text out of the html

        text =  soup.get_text()

        # Returning text stripping out all uris

        return re.sub(uri_re, "", text)

    else:

        return ""
for df in dataframes.values():

    df["content"] = df["content"].map(stripTagsAndUris)
corpus = []

vectorizer = CountVectorizer(min_df=1)

for df in dataframes.values():

    for title in df['title']:

        corpus.append(title)

    for content in df['content']:

        corpus.append(content)

X = vectorizer.fit_transform(corpus)
analyzer = vectorizer.build_analyzer()

tokenizer = vectorizer.build_tokenizer()
for df in dataframes.values():

    df["title"] = df["title"].map(analyzer)

    df["content"] = df["content"].map(analyzer)

    df["tags"] = df["tags"].map(tokenizer)
dataframes['cooking'].head()