import numpy as np 

import pandas as pd 

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
data = pd.read_csv('/kaggle/input/uncover/UNCOVER/HDE/acaps-covid-19-government-measures-dataset.csv')

data.head()
data.isnull().sum()
countries = data['country'].value_counts()

countries
countries = countries.to_frame().reset_index().rename(columns={"index": "Country", "country": "mentions"})

countries.head()
import plotly.express as px



fig = px.choropleth(countries, locations="Country", locationmode='country names', 

                  color="mentions", hover_name="Country", 

                  title="mentions of governmetn measures", hover_data=["mentions"])

fig.show()
data['category'].value_counts()
fig = px.pie(data['category'].value_counts(), 

             values = data['category'].value_counts(), 

             names=data['category'].value_counts().index,

            title = "Categories of the measures")

fig.show()
dummies = pd.get_dummies(data['category'])

dummies.head()
data = pd.concat([data, dummies], axis=1, sort=False)

data.head()
stat = data['measure'].value_counts()

stat.head()
fig = px.pie(stat, 

             values = stat, 

             names=stat.index,

            title = "Measures")

fig.show()
data['measure'] = data['measure'].astype('category')

data.dtypes
data['measure_cat'] = data['measure'].cat.codes

data.head()
len(data['comments'].value_counts())
data['comments'].fillna('',inplace=True)
data['comments_modified'] = data['comments'].str.strip('.,;:""')

data['comments_modified'] = data['comments_modified'].str.lower()
data['comments_modified'] = data['comments_modified'].str.translate({ord(k): None for k in '0123456789'})
data['comments_modified'].head()
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfTransformer

from itertools import islice
vectorizer = CountVectorizer(stop_words='english', min_df=2, max_df=0.9, ngram_range=(1,1))

vectorizer
vectorizer.fit(data['comments_modified'])
list(islice(vectorizer.vocabulary_.items(), 20))
len(vectorizer.vocabulary_)
bag_of_words = vectorizer.transform(data['comments_modified'])

print( "shape ", bag_of_words.toarray().shape)

print('sparsity: %.2f%%' % (100.0 * bag_of_words.nnz / (bag_of_words.shape[0] * bag_of_words.shape[1])))
occ = np.asarray(bag_of_words.sum(axis=0)).ravel().tolist()

counts_df = pd.DataFrame({'term': vectorizer.get_feature_names(), 'occurrences': occ})

counts_df.sort_values(by='occurrences', ascending=False).head(20)
transformer = TfidfTransformer()

transformed_weights = transformer.fit_transform(bag_of_words)

transformed_weights.shape
importance = transformed_weights.toarray().mean(axis=0)

im_df = pd.DataFrame({'word': vectorizer.get_feature_names(), 'importance': importance})

im_df.sort_values(by='importance', ascending=False).head(15)
from sklearn.feature_extraction.text import TfidfVectorizer



vectorizer1 = TfidfVectorizer(stop_words='english', min_df=2, max_df=0.9, ngram_range=(2,3))

weights_n_gram = vectorizer1.fit_transform(data['comments_modified'])

importance_n_gram = weights_n_gram.toarray().mean(axis=0)

im_df_n_gram = pd.DataFrame({'word': vectorizer1.get_feature_names(), 'importance': importance_n_gram})

im_df_n_gram.sort_values(by='importance', ascending=False).head(20)
features = ['state of emergency', 

            'non essential', 

            'public gatherings', 

            'self quarantine',

            'international flights',

            'stay home',

            'schools closed ',

            'social distancing'] #we added some stop words and modifications back to the text, so we can find them in original dataset
for feature in features:

    data[feature] = data['comments_modified'].apply(lambda x: 1 if feature in x else 0)
data.head()
for feature in features:

    print(feature, ' is in ', data[feature].value_counts()[1], ' rows')

    