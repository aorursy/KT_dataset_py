import re

import pandas as pd

import numpy as np

from nltk.corpus import stopwords

from nltk import stem

from sklearn.externals.joblib import Parallel, delayed
def tanimoto_coefficient(words1, words2):

    try:

        res = len(words1 & words2) / (len(words1) + len(words2) - len(words1 & words2))

    except ZeroDivisionError:

        res = 0

    return res
stops = set(stopwords.words("english"))

stemmer = stem.LancasterStemmer()

lemmatizer = stem.wordnet.WordNetLemmatizer()

def st2words(st):

    st = str(st)

    st = re.sub(r'\?|\.|\,|\(|\)|－|\'|\"', " ", st)

    words = [w for w in st.split() if w != ""]

    words = [stemmer.stem(w) if w != w.upper() else w for w in words]

    words = [w.lower()  for w in st.split()]

    words = [lemmatizer.lemmatize(w) for w in words]

    words = [w for w in words if w not in stops]

    return set(words)
def tanimoto_coefficient_from_st(st1, st2):

    words1 = st2words(st1)

    words2 = st2words(st2)

    return tanimoto_coefficient(words1, words2)
train = pd.read_csv("../input/train.csv")
train["R"] = Parallel(n_jobs=-1, verbose=2)([delayed(tanimoto_coefficient_from_st)(row[0], row[1]) for row in train[["question1", "question2"]].values])
train[train.is_duplicate == 0].sort(["R"], ascending=False)