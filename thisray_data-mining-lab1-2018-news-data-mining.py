# necessary for when working with external scripts
%load_ext autoreload
%autoreload 2
## this block is only for kaggle to setup the directory
## and remember to turn on the Internet connected setting!
import os
os.chdir("../input/")
print(os.listdir("../input"))
print(os.listdir("../input/helpers"))
# categories
categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']
# obtain the documents containing the categories provided
from sklearn.datasets import fetch_20newsgroups

twenty_train = fetch_20newsgroups(subset='train', categories=categories, \
                                  shuffle=True, random_state=42)
twenty_train.data[0:2]
twenty_train.target_names
len(twenty_train.data)
len(twenty_train.filenames)
# An example of what the subset contains
print("\n".join(twenty_train.data[0].split("\n")))
print(twenty_train.target_names[twenty_train.target[0]])
# category of first 10 documents.
twenty_train.target[:10]
for t in twenty_train.target[:10]:
    print(twenty_train.target_names[t])
# Answer here

twenty_train.data[0:2]
twenty_train.target
import pandas as pd

# my functions
import helpers.data_mining_helpers as dmh

# construct dataframe from a list
X = pd.DataFrame.from_records(dmh.format_rows(twenty_train), columns= ['text'])
len(X)
X[0:2]
# add category to the dataframe
X['category'] = twenty_train.target
# add category label also
X['category_name'] = X.category.apply(lambda t: dmh.format_labels(t, twenty_train))
X[0:10]
# a simple query
X[0:10][["text", "category_name"]]
X[-10:]
# using loc (by position)
X.iloc[::10, :][0:10]
# using loc (by label)
X.loc[::10, 'text'][0:10]
# standard query (Cannot simultaneously select rows and columns)
X[::10][0:10]
# Answer here

# Answer here

X.isnull()
X.isnull().apply(lambda x: dmh.check_missing_values(x))
# Answer here

dummy_series = pd.Series(["dummy_record", 1], index=["text", "category"])
dummy_series
result_with_series = X.append(dummy_series, ignore_index=True)
# check if the records was commited into result
len(result_with_series)
result_with_series.isnull().apply(lambda x: dmh.check_missing_values(x))
# dummy record as dictionary format
dummy_dict = [{'text': 'dummy_record',
               'category': 1
              }]
X = X.append(dummy_dict, ignore_index=True)
len(X)
X.isnull().apply(lambda x: dmh.check_missing_values(x))
X.dropna(inplace=True)
X.isnull().apply(lambda x: dmh.check_missing_values(x))
len(X)
import numpy as np

NA_dict = [{ 'id': 'A', 'missing_example': np.nan },
           { 'id': 'B'                    },
           { 'id': 'C', 'missing_example': 'NaN'  },
           { 'id': 'D', 'missing_example': 'None' },
           { 'id': 'E', 'missing_example':  None  },
           { 'id': 'F', 'missing_example': ''     }]

NA_df = pd.DataFrame(NA_dict, columns = ['id','missing_example'])
NA_df
NA_df['missing_example'].isnull()
# Answer here

X.duplicated()
sum(X.duplicated())
sum(X.duplicated('text'))
dummy_duplicate_dict = [{
                             'text': 'dummy record',
                             'category': 1, 
                             'category_name': "dummy category"
                        },
                        {
                             'text': 'dummy record',
                             'category': 1, 
                             'category_name': "dummy category"
                        }]
X = X.append(dummy_duplicate_dict, ignore_index=True)
len(X)
sum(X.duplicated('text'))
X.drop_duplicates(keep=False, inplace=True) # inplace applies changes directly on our dataframe
len(X)
X_sample = X.sample(n=1000)
len(X_sample)
X_sample[0:4]
# Answer here

import matplotlib.pyplot as plt
%matplotlib inline
categories
print(X.category_name.value_counts())

# plot barchart for X_sample
X.category_name.value_counts().plot(kind = 'bar',
                                    title = 'Category distribution',
                                    ylim = [0, 650],        
                                    rot = 0, fontsize = 11, figsize = (8,3))
print(X_sample.category_name.value_counts())

# plot barchart for X_sample
X_sample.category_name.value_counts().plot(kind = 'bar',
                                           title = 'Category distribution',
                                           ylim = [0, 300], 
                                           rot = 0, fontsize = 12, figsize = (8,3))
# Answer here

# Answer here

import nltk
# takes a like a minute or two to process
X['unigrams'] = X['text'].apply(lambda x: dmh.tokenize_text(x))
X[0:4]["unigrams"]
X[0:4]
list(X[0:1]['unigrams'])
from sklearn.feature_extraction.text import CountVectorizer

count_vect = CountVectorizer()
X_counts = count_vect.fit_transform(X.text)
X_counts
analyze = count_vect.build_analyzer()
analyze("Hello World!")
#" ".join(list(X[4:5].text))
# Answer here

# We can check the shape of this matrix by:
X_counts.shape
# We can obtain the feature names of the vectorizer, i.e., the terms
# usually on the horizontal axis
count_vect.get_feature_names()[0:10]
X[0:5]
# we convert from sparse array to normal array
X_counts[0:5,0:100].toarray()
# Answer here

count_vect.transform(['Something completely new.']).toarray()
count_vect.transform(['00 Something completely new.']).toarray()
# first twenty features only
plot_x = ["term_"+str(i) for i in count_vect.get_feature_names()[0:20]]
plot_x
# obtain document index
plot_y = ["doc_"+ str(i) for i in list(X.index)[0:20]]
plot_z = X_counts[0:20, 0:20].toarray()
import seaborn as sns

df_todraw = pd.DataFrame(plot_z, columns = plot_x, index = plot_y)
plt.subplots(figsize=(9, 7))
ax = sns.heatmap(df_todraw,
                 cmap="PuRd",
                 vmin=0, vmax=1, annot=True)
# Answer here

from sklearn.decomposition import PCA
X_reduced = PCA(n_components = 2).fit_transform(X_counts.toarray())
X_reduced.shape
categories
col = ['coral', 'blue', 'black', 'm']

# plot
fig = plt.figure(figsize = (25,10))
ax = fig.subplots()

for c, category in zip(col, categories):
    xs = X_reduced[X['category_name'] == category].T[0]
    ys = X_reduced[X['category_name'] == category].T[1]
   
    ax.scatter(xs, ys, c = c, marker='o')

ax.grid(color='gray', linestyle=':', linewidth=2, alpha=0.2)
ax.set_xlabel('\nX Label')
ax.set_ylabel('\nY Label')

plt.show()
# Answer here

# note this takes time to compute. You may want to reduce the amount of terms you want to compute frequencies for
term_frequencies = []
for j in range(0,X_counts.shape[1]):
    term_frequencies.append(sum(X_counts[:,j].toarray()))
term_frequencies = np.asarray(X_counts.sum(axis=0))[0]
term_frequencies[0]
plt.subplots(figsize=(100, 10))
g = sns.barplot(x=count_vect.get_feature_names()[:300], 
            y=term_frequencies[:300])
g.set_xticklabels(count_vect.get_feature_names()[:300], rotation = 90);
# Answer here

# Answer here

# Answer here

import math
term_frequencies_log = [math.log(i) for i in term_frequencies]
plt.subplots(figsize=(100, 10))
g = sns.barplot(x=count_vect.get_feature_names()[:300],
                y=term_frequencies_log[:300])
g.set_xticklabels(count_vect.get_feature_names()[:300], rotation = 90);
from sklearn import preprocessing, metrics, decomposition, pipeline, dummy
mlb = preprocessing.LabelBinarizer()
mlb.fit(X.category)
mlb.classes_
X['bin_category'] = mlb.transform(X['category']).tolist()
X[0:9]
# Answer here
