# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import pandas as pd

import numpy as np



import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.feature_extraction.text import CountVectorizer



%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
df_arxive = pd.read_csv("../input/scirate_quant-ph_2016.csv")

df_arxive = df_arxive.drop("Unnamed: 0", axis=1)
abstract_corpus = []

abstracts = df_arxive.title.values

for abstract in abstracts:

    abstract_corpus.append(abstract)



vectorizer = CountVectorizer(stop_words='english')

vectorizer.fit(abstract_corpus)



X = vectorizer.transform(abstract_corpus)
cites = df_arxive.scites.values



cites_modified = []

for i in range(X.shape[0]):

    cites_modified.append(X[i,:]*cites[i])



cites_modified = np.asarray(cites_modified)



sums = np.zeros((1,X.shape[1]))

for title in cites_modified:

    sums += title
titles_sum_frame = pd.DataFrame(sums[0,:].T,index=vectorizer.get_feature_names(),columns=["title_count"])

titles_sum_frame = titles_sum_frame.sort('title_count',ascending=False)

titles_sum_frame = titles_sum_frame.drop('quantum')

titles_sum_frame.head(20).plot.barh()

plt.title("Hot So")

titles_sum_frame.tail(20).plot.barh()

plt.title("Not So")
abstract_corpus = []

abstracts = df_arxive.abstract.values

for abstract in abstracts:

    abstract_corpus.append(abstract)



vectorizer = CountVectorizer(stop_words='english')

vectorizer.fit(abstract_corpus)



X = vectorizer.transform(abstract_corpus)
cites = df_arxive.scites.values



cites_modified = []

for i in range(X.shape[0]):

    cites_modified.append(X[i,:]*cites[i])



cites_modified = np.asarray(cites_modified)



sums = np.zeros((1,X.shape[1]))

for title in cites_modified:

    sums += title
abstract_sum_frame = pd.DataFrame(sums[0,:].T,index=vectorizer.get_feature_names(),columns=["abstract_count"])

abstract_sum_frame = abstract_sum_frame.sort('abstract_count',ascending=False)

abstract_sum_frame = abstract_sum_frame.drop('quantum')

abstract_sum_frame.head(20).plot.barh()

plt.title("Hot So")

abstract_sum_frame.tail(20).plot.barh()

plt.title("Not So")
total_frame = pd.DataFrame.join(abstract_sum_frame,titles_sum_frame)

total_frame['total_count'] = total_frame.abstract_count + total_frame.title_count

total_frame = total_frame.sort("total_count",ascending=False)
title_not_so = set(titles_sum_frame.tail(300).index)

abstract_not_so = set(abstract_sum_frame.tail(300).index)

intersection_not_so = set.intersection(title_not_so,abstract_not_so)

intersection_not_so
title_hot_so = set(titles_sum_frame.head(50).index)

abstract_hot_so = set(abstract_sum_frame.head(50).index)

intersection_hot_so = set.intersection(title_hot_so,abstract_hot_so)

intersection_hot_so
total_frame.ix[list(intersection_hot_so)].sort('title_count').plot.barh()

plt.title("Hot So Global View")

titles_sum_frame.ix[list(intersection_hot_so)].sort('title_count').plot.barh()

plt.title("Hot So Topics")

total_frame.ix[list(intersection_not_so)].sort('title_count').plot.barh()

plt.title("Not So")