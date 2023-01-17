# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split



train_data = pd.read_table('../input/train.tsv')

test_data = pd.read_table('../input/test.tsv')

# Any results you write to the current directory are saved as output.
def separate_and_fill(data):

    if 'price' in data:

        y_data = np.log1p(data['price'].copy())

        data = data.drop('price', axis=1)

    else:

        y_data = None

    data['category_name'] = data['category_name'].fillna('NA/NA/NA').astype(str)

    data['cat0'] = data['category_name'].map(lambda x: x.split('/')[0])

    data['cat1'] = data['category_name'].map(lambda x: x.split('/')[1])

    data['cat2'] = data['category_name'].map(lambda x: x.split('/')[2])

    data.drop('category_name', axis=1, inplace=True)

    data['brand_name'] = data['brand_name'].fillna('Missing').astype(str).map(lambda x: x.lower())

    data['shipping'] = data['shipping'].astype(int)

    data['name'] = data['name'].astype(str).map(lambda x: x.lower())

    data['item_condition_id'] = data['item_condition_id'].astype(str)

    data['item_description'] = data['item_description'].fillna('Missing')

    return data, y_data
from sklearn.preprocessing import OneHotEncoder, LabelEncoder



from sklearn.pipeline import Pipeline, FeatureUnion

from sklearn.base import BaseEstimator, TransformerMixin



from sklearn.linear_model import LinearRegression, Ridge



from sklearn.feature_extraction.text import TfidfVectorizer



from sklearn.decomposition import TruncatedSVD
class DataFrameLabelEncoder(BaseEstimator, TransformerMixin):

    def __init__(self, keys):

        self.keys = keys

        

    def fit(self, X, y=None):

        return self

    

    def transform(self, X, y=None):

        label_encoder = LabelEncoder()

        for key in self.keys:

            X[key] = label_encoder.fit_transform(X[key])

        return X

    

    

class ItemSelector(BaseEstimator, TransformerMixin):

    def __init__(self, key):

        self.key = key



    def fit(self, x, y=None):

        return self



    def transform(self, data_dict):

        return data_dict[self.key]
from nltk.corpus import stopwords

import re



stopwords = {x: 1 for x in stopwords.words('english')}

non_alphanums = re.compile(u'[^A-Za-z0-9]+')





def normalize_text(text):

    return u" ".join(

        [x for x in [y for y in non_alphanums.sub(' ', text).lower().strip().split(" ")] \

         if len(x) > 1 and x not in stopwords])
X_train, y_train = separate_and_fill(train_data)

X_test, _ = separate_and_fill(test_data)
pipe = Pipeline([

    ('union', FeatureUnion(

        transformer_list=[

            ('description_tdfidf', Pipeline([

                ('selector', ItemSelector(key='item_description')),

                ('tfidf', TfidfVectorizer(max_df=0.95, min_df=2, max_features=5000,

                                   stop_words='english', ngram_range=(1, 3))),

                ('svd', TruncatedSVD(n_components = 100)),

            ])),

            ('the_rest_of_the_data_frame', Pipeline([

                ('selector_2', ItemSelector(

                    ['item_condition_id', 'cat0', 'cat1', 'cat2', 'brand_name', 'shipping'])),

                ('encode_cat', DataFrameLabelEncoder(['cat0', 'cat1', 'cat2', 'brand_name'])),

                ('onehot_encode', OneHotEncoder(categorical_features=[1, 2, 3, 4])),                

            ]))

        ]

    )

)])
num_train_samples = len(X_train)
X_all = pd.concat([X_train, X_test])
X_all_encoded = pipe.fit_transform(X_all, None)
X_train = X_all_encoded[:num_train_samples,:]

X_test = X_all_encoded[num_train_samples:,:]
lin_reg = Ridge()

lin_reg.fit(X_train, y_train)
y_pred = lin_reg.predict(X_test)
y_pred = np.maximum(0, np.expm1(y_pred))
df = pd.DataFrame(y_pred, columns=['price'])

df.to_csv('./submission.csv', index_label='test_id')    