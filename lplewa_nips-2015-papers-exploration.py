%matplotlib inline
from IPython.display import HTML
import pandas as pd
import re
import sqlite3

con = sqlite3.connect('../input/database.sqlite')

papers = pd.read_sql_query("""
SELECT *
FROM Papers""", con)
papers["Keywords"] = ""
papers["Paper"] = ""

for i in range(len(papers)):
    papers.loc[i, "Paper"] = "<" + "a href='https://papers.nips.cc/paper/" + papers["PdfName"][i] + "'>" + papers["Title"][i] + "<" + "/a>"
    papers.loc[i, "Keywords"] = papers["Abstract"][i]

# papers = papers.sort_values("Keywords", ascending=False)
# papers.index = range(1, len(papers)+1)
# pd.set_option("display.max_colwidth", -1)

# HTML(papers[["Paper", "Keywords"]].to_html(escape=False))
import operator
# import nltk
# from nltk.corpus import stopwords

# TODO: collect a machine learning algorithm dataset instead of manually inputting keywords

ML_KEYWORDS = [
    "neural", "stochastic", "networks", "linear", "gradient", "convolutional",
    "markov", "bayesian", "bayes", "decision", "tree", "vector", "SVM", "support",
    "genetic", "gaussian", "k-neighbors", "neighbor", "forests", "classifier", "regression",
    "boosting"
]

word_freq = dict()

for i in range(len(papers)):
    words = papers["Keywords"][i].split(" ")
    for word in words:
        word = word.lower()
        if word not in ML_KEYWORDS:
            pass
        elif word in word_freq:
            word_freq[word] = word_freq[word] + 1
        else:
            word_freq[word] = 1

sorted_by_frequency = sorted(word_freq.items(), key=operator.itemgetter(1), reverse=True)
print("Frequencies of machine learning keywords")
print(sorted_by_frequency)
import numpy as np
from matplotlib import pyplot as plt

x_values = [unit[0] for unit in sorted_by_frequency]
y_values = [unit[1] for unit in sorted_by_frequency]

fig = plt.figure()

width = .5
ind = np.arange(len(y_values))
plt.bar(ind, y_values)
plt.xticks(ind + width / 2, x_values)

fig.autofmt_xdate()

plt.savefig("figure.pdf")