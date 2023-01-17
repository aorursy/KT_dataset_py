# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import warnings

warnings.filterwarnings('ignore')



train_ds = pd.read_csv('/kaggle/input/umich-si650-nlp/train.csv')

train_ds.head(5)
# if the texts are truncated, just in case lets increase the max width to get the full text.

pd.set_option('max_colwidth',800)

print(train_ds[train_ds.label==1][0:5]) #positive_comments

print(train_ds[train_ds.label==0][0:5]) #negetive_comments
train_ds.shape

# data has 5668 rows with 2 columns.
train_ds.info()

#We can see there are no missing values
# count plot to check the positive and negeticve comments ratio

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



plt.figure(figsize=(6,5))

ax = sns.countplot(x='label',data=train_ds)

for p in ax.patches:

    ax.annotate(p.get_height(), (p.get_x()+0.1,p.get_height()+50))
from sklearn.feature_extraction.text import CountVectorizer

Count_Vectorizer = CountVectorizer()

feature_vector = Count_Vectorizer.fit(train_ds.sentence)

#get the feature names

features = feature_vector.get_feature_names()

print('Total number of features :',len(features))
# Lets look at some of the random samples

import random

random.sample(features,10)
train_ds_features= Count_Vectorizer.transform(train_ds.sentence)            

type(train_ds_features)

train_ds_features.shape

# This matrix or data frame obtained consists of all non-zeros and zero values and hence lets calculate the ratio of non-zeros to zeros to see where we stand 

# `Lets store it as an sparse matrix, i.e this matrix stores only the vectors whose values are 1.
# To get number of non_zeros

train_ds_features.getnnz()
print('Density of matrix:',train_ds_features.getnnz() *100 /( train_ds_features.shape[0] * train_ds_features.shape[1]))
# Now lets visualize these count vectors, by converting the matrix into a dataFrame and lets set column namres as actual feature names

# Converting matrix into a data Frame

train_ds_df = pd.DataFrame(train_ds_features.todense())

train_ds_df.columns = features

train_ds[0:1]

train_ds_df.iloc[0:1,230:250]
train_ds_df[[ 'brokeback', 'mountain', 'is', 'such', 'horrible', 'movie']][0:1]

# We can see al of them are correcty named as 1
# Summing the occurance of features column wise

features_counts = np.sum(train_ds_features.toarray(),axis=0)

feature_counts_df = pd.DataFrame(dict(features=features,counts=features_counts))

feature_counts_df
plt.figure(figsize=(12,5))

plt.hist(feature_counts_df.counts, bins=50, range = (0,2000));

plt.xlabel('Frequency of words')

plt.ylabel('Density')

       
# Fromm the graph we can observe that 1136 words are present only once, these can be ignored, we can set the max_features count depending the size of the file you are working

# on, for now lets take it as 1000

len(feature_counts_df[feature_counts_df.counts == 1])
count_vectorizer = CountVectorizer(max_features=1000)

feature_vector=count_vectorizer.fit(train_ds.sentence)

features = feature_vector.get_feature_names()

train_ds_features=count_vectorizer.transform(train_ds.sentence)

# count the freq of features

feature_counts=np.sum(train_ds_features.toarray(),axis=0)

feature_counts=pd.DataFrame(dict(features = features, counts = feature_counts))
feature_counts.sort_values('counts',ascending=False)[0:15]
from sklearn.feature_extraction import text 

my_stop_words = text.ENGLISH_STOP_WORDS



print('Few pre-defined stop words are:',list(my_stop_words)[0:10])
# We can also add our own stop words to the list.

my_stop_words = text.ENGLISH_STOP_WORDS.union(['harry','potter','da','vinci','code','mountain','movie','movies'])
# Creating count vectors without stop words

count_vectorizer = CountVectorizer(stop_words = my_stop_words,max_features=1000)

feature_vector=count_vectorizer.fit(train_ds.sentence)

features = feature_vector.get_feature_names()

train_ds_features=count_vectorizer.transform(train_ds.sentence)

# count the freq of features

feature_counts=np.sum(train_ds_features.toarray(),axis=0)

feature_counts=pd.DataFrame(dict(features = features, counts = feature_counts))
feature_counts.sort_values('counts',ascending=False)[0:15]
from nltk.stem.snowball import PorterStemmer

stemmer = PorterStemmer()

analyzer = CountVectorizer().build_analyzer()



# custom function for stemming and stop word removal

def stemmed_words(doc):

#stemming of words

    stemmed_words=[stemmer.stem(w) for w in analyzer(doc)]

# removing these words in stop words list

    non_stop_words= [word for word in stemmed_words if not word in my_stop_words]

    return non_stop_words

    



       
count_vectorizer = CountVectorizer(analyzer = stemmed_words,max_features=1000)

feature_vector=count_vectorizer.fit(train_ds.sentence)

train_ds_features=count_vectorizer.transform(train_ds.sentence)

feature_counts=np.sum(train_ds_features.toarray(),axis=0)

feature_counts=pd.DataFrame(dict(features = features, counts = feature_counts))

feature_counts.sort_values('counts',ascending=False)[0:15]
# Now we can see that all the words are reduced to there root words. Next lets classify them into sentiments.

train_ds_df = pd.DataFrame(train_ds_features.todense())

train_ds_df.columns = features

train_ds_df['sentiment']= train_ds.label
sns.barplot(x='sentiment', y = 'hate', data = train_ds_df, estimator=sum)