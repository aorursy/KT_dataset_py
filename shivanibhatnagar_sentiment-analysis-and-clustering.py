# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input/"]).decode("utf8"))



# Any results you write to the current directory are saved as output.

"../input/NAME_OF_DATAFILE.csv"
train_ds = pd.read_csv( "../input/dictionary-for-sentiment-analysis/dict.csv", 

                       delimiter=",", encoding = "ISO-8859-1",

                       names = ["Text", "Sentiment"] )
train_ds.head(10)
train_ds['Sentiment'] = train_ds['Sentiment'].map({'positive': 1, 'negative': 0})
train_ds.head(10)
from sklearn.feature_extraction.text import CountVectorizer
count_vectorizer = CountVectorizer( max_features = 5000 )
feature_vector = count_vectorizer.fit( train_ds.Text )

train_ds_features = count_vectorizer.transform( train_ds.Text )

features = feature_vector.get_feature_names()
features[0:20]
features_counts = np.sum( train_ds_features.toarray(), axis = 0 )
feature_counts = pd.DataFrame( dict( features = features,

                                  counts = features_counts ) )
feature_counts.head(5)
count_vectorizer = CountVectorizer( stop_words = "english",

                                 max_features = 5000 )

feature_vector = count_vectorizer.fit( train_ds.Text )

train_ds_features = count_vectorizer.transform( train_ds.Text )
features = feature_vector.get_feature_names()

features_counts = np.sum( train_ds_features.toarray(), axis = 0 )

feature_counts = pd.DataFrame( dict( features = features,

                                  counts = features_counts ) )

feature_counts.sort_values( "counts", ascending = False )[0:20]
test_ds = pd.read_csv( "../input/million-headlines/abcnews-date-text.csv", 

                       delimiter=",", encoding = "ISO-8859-1")
test_ds.head(10)
test_ds.count()
test_ds.dropna()
test_ds.count()
headline_text = count_vectorizer.transform( test_ds.headline_t )
headline_text = count_vectorizer.transform( test_ds.headline_text.astype('U') )
headline_text[1]
from sklearn.naive_bayes import GaussianNB

from sklearn.cross_validation import train_test_split

clf = GaussianNB()
train_x, test_x, train_y, test_y = train_test_split( train_ds_features,

                                                  train_ds.Sentiment,

                                                  test_size = 0.3,

                                                  random_state = 42 )
clf.fit( train_x.toarray(), train_y )
test_ds_predicted = clf.predict( test_x.toarray() )
from sklearn import metrics

cm = metrics.confusion_matrix( test_y, test_ds_predicted )
cm
import matplotlib as plt

import seaborn as sn

%matplotlib 

sn.heatmap(cm, annot=True,  fmt='.2f' )
score = metrics.accuracy_score( test_y, test_ds_predicted )
score
test_ds["sentiment"] = clf.predict( headline_text.toarray() )
test_ds[0:100]