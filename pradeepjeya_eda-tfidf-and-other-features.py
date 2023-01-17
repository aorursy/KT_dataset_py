# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt

% matplotlib inline

import seaborn as sns

color = sns.color_palette()

sns.set(style="whitegrid", color_codes=True)

sns.set(font_scale=1)





import plotly.plotly as py

import plotly.graph_objs as go

from plotly import tools



from plotly.offline import download_plotlyjs, init_notebook_mode, iplot

init_notebook_mode(connected=True)
# Reading the json train and test files



train = pd.read_json("../input/train.json")

test = pd.read_json("../input/test.json")
train.describe() # for numerical features
int_level = train['interest_level'].value_counts()



plt.figure(figsize=(8,4))

sns.barplot(int_level.index, int_level.values, alpha=0.8, color=color[1])

plt.ylabel('Number of Occurrences', fontsize=12)

plt.xlabel('Interest level', fontsize=12)

plt.show()
# Average price of property by interest level





int_level_price = train['price'].groupby(train['interest_level']).mean()

int_level_bath = train['bathrooms'].groupby(train['interest_level']).mean()

int_level_bed = train['bedrooms'].groupby(train['interest_level']).mean()







plt.figure(figsize=(8,4))

sns.barplot(int_level_price.index, int_level_price.values, alpha=0.8, color=color[2])

plt.ylabel(' Mean Price', fontsize=12)

plt.xlabel('Interest level', fontsize=12)

plt.show()







plt.figure(figsize=(8,4))

sns.barplot(int_level_bath.index, int_level_bath.values, alpha=0.8, color=color[4])

plt.ylabel(' Avg No of Bathrooms', fontsize=12)

plt.xlabel('Interest level', fontsize=12)

plt.show()









plt.figure(figsize=(8,4))

sns.barplot(int_level_bath.index, int_level_bath.values, alpha=0.8, color=color[5])

plt.ylabel('Avg No of Bedrooms', fontsize=12)

plt.xlabel('Interest level', fontsize=12)

plt.show()
from sklearn import preprocessing



lbl = preprocessing.LabelEncoder()

lbl.fit(list(train['manager_id'].values))

train['manager_id'] = lbl.transform(list(train['manager_id'].values))



temp = pd.get_dummies(train.interest_level)

temp = pd.concat([train.manager_id, temp], axis=1).groupby(train['manager_id']).mean()

temp.columns = ['manager_id','high_frac','low_frac','medium_frac']

temp['manager_skill'] = temp['high_frac']*2 + temp['medium_frac'] + temp['low_frac']*0.2

temp.index = temp.manager_id

del temp['manager_id']
temp.head(3)
# Merging manager skill with training set



train = train.merge(temp.reset_index(), how='left', left_on='manager_id', right_on = 'manager_id')

train.head(3)
train['feat_len'] = train['features'].map(lambda text: len(text))

train.feat_len.plot(bins=20, kind='hist')
import nltk

import string

import os



from sklearn.feature_extraction.text import TfidfVectorizer

from nltk.stem.porter import PorterStemmer

from sklearn.feature_extraction.text import CountVectorizer

from textblob import TextBlob as tb

from nltk.corpus import stopwords

from nltk.stem.wordnet import WordNetLemmatizer

import nltk





%matplotlib inline

import matplotlib.pyplot as plt

import csv
columns = ['new_features','new_feat_lem']

df = pd.DataFrame(index=train.index, columns = columns)

for i in range(len(train)):

    df.new_features.iloc[i] = ','.join(map(str,train.features.iloc[i]))
train = train.join(df.new_features)

train['new_features'] = train['new_features'].str.lower()

lemmatizer = WordNetLemmatizer()

for i,w in enumerate(train.new_features):

    df.new_feat_lem.iloc[i] = lemmatizer.lemmatize(w)
train_new = train.join(df.new_feat_lem)

vectorizer = TfidfVectorizer(stop_words='english',min_df=0.01,strip_accents = ascii,norm='l2')

transformed = vectorizer.fit_transform(train_new['new_feat_lem']).toarray()

print("Num words:", len(vectorizer.get_feature_names()))
df2 = pd.DataFrame(transformed, index=train_new.index,columns=vectorizer.get_feature_names())

train_new = pd.concat([train_new, df2], axis=1, join_axes=[train_new.index])
# Checking if all building id's are unique

len(train_new.building_id.unique())
# Creating a new feature that counts the number of times a building ID appears



columns = ['No_of_listings_per_build_id']

df2 = pd.DataFrame(columns = columns)

df2['No_of_listings_per_build_id']= train_new.building_id.value_counts()

df2 = df2.reset_index()

columns = {'index': 'building_id'}

df2.rename(columns = columns, inplace=True)

df2.head(3)
# Joining with training set

train_new = train_new.merge(df2.reset_index(), how='left', left_on='building_id', right_on = 'building_id')

del train_new['index']

train_new.head(3)
