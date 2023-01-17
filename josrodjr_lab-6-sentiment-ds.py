import numpy as np 

import pandas as pd 

pd.set_option('display.max_colwidth', -1)

from time import time

import re

import string

import os

import emoji

from pprint import pprint

import collections



import matplotlib.pyplot as plt

import seaborn as sns

sns.set(style="darkgrid")

sns.set(font_scale=1.3)



from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import train_test_split

from sklearn.pipeline import Pipeline, FeatureUnion

from sklearn.metrics import classification_report



from sklearn.naive_bayes import MultinomialNB

from sklearn.linear_model import LogisticRegression

from sklearn.externals import joblib



import gensim



from nltk.corpus import stopwords

from nltk.stem import PorterStemmer

from nltk.tokenize import word_tokenize



import warnings

warnings.filterwarnings('ignore')



np.random.seed(37)
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/grammar-and-online-product-reviews/GrammarandProductReviews.csv', header=0)

df = df.reindex(np.random.permutation(df.index))
sns.factorplot(x="reviews.doRecommend", data=df, kind="count", size=6, aspect=1.5, palette="PuBuGn_d")

plt.show();
sns.factorplot(x="reviews.rating", data=df, kind="count", size=6, aspect=1.5, palette="PuBuGn_d")

plt.show();
the_df = df[['name' , 'reviews.rating']]

the_df = the_df.rename(columns={ "reviews.rating": "rating"})



grouped = the_df.groupby('name').sum().reset_index()

best = grouped.sort_values('rating', ascending=False)

best.head(10)
grouped = the_df.groupby('name').sum().reset_index()

worst = grouped.sort_values('rating', ascending=True)

worst.head(10)
the_df = df[['reviews.username' , 'reviews.rating']]

the_df = the_df.rename(columns={"reviews.username":"username" ,"reviews.rating": "rating"})



#grouped = the_df.groupby('username').sum().reset_index()

grouped = the_df.groupby('username').filter(lambda group: group.size < 3)

best = grouped.sort_values('rating', ascending=False)

best.head(10)
the_df = df[['manufacturer' , 'reviews.rating']]

the_df = the_df.rename(columns={"reviews.rating": "rating"})



grouped = the_df.groupby('manufacturer').sum().reset_index()

worst = grouped.sort_values('rating', ascending=True)

worst.head(10)
the_df = df[['manufacturer' , 'reviews.rating']]

the_df = the_df.rename(columns={"reviews.rating": "rating"})



grouped = the_df.groupby('manufacturer').sum().reset_index()

best = grouped.sort_values('rating', ascending=False)

best.head(10)
class TextCounts(BaseEstimator, TransformerMixin):

    

    def count_regex(self, pattern, tweet):

        return len(re.findall(pattern, tweet))

    

    def fit(self, X, y=None, **fit_params):

        # fit method is used when specific operations need to be done on the train data, but not on the test data

        return self

    

    def transform(self, X, **transform_params):

        count_words = X.apply(lambda x: self.count_regex(r'\w+', x)) 

        count_mentions = X.apply(lambda x: self.count_regex(r'@\w+', x))

        count_hashtags = X.apply(lambda x: self.count_regex(r'#\w+', x))

        count_capital_words = X.apply(lambda x: self.count_regex(r'\b[A-Z]{2,}\b', x))

        count_excl_quest_marks = X.apply(lambda x: self.count_regex(r'!|\?', x))

        count_urls = X.apply(lambda x: self.count_regex(r'http.?://[^\s]+[\s]?', x))

        # We will replace the emoji symbols with a description, which makes using a regex for counting easier

        # Moreover, it will result in having more words in the tweet

        count_emojis = X.apply(lambda x: emoji.demojize(x)).apply(lambda x: self.count_regex(r':[a-z_&]+:', x))

        

        df = pd.DataFrame({'count_words': count_words

                           , 'count_mentions': count_mentions

                           , 'count_hashtags': count_hashtags

                           , 'count_capital_words': count_capital_words

                           , 'count_excl_quest_marks': count_excl_quest_marks

                           , 'count_urls': count_urls

                           , 'count_emojis': count_emojis

                          })

        

        return df
#the only things we are going to use in the dataset is reviews.rating and reviews.text

new_df = df[['reviews.text', 'reviews.rating']]

new_df = new_df.rename(columns={"reviews.text": "text", "reviews.rating": "rating"})

#new_df

# remove all non letters from column text

new_df.text = new_df.text.str.replace('[^a-zA-Z]', ' ')

#try and parse them as string

new_df['text'] = new_df['text'].astype(str)

#new_df.keys()



tc = TextCounts()

df_eda = tc.fit_transform(new_df.text)
df_eda['rating'] = new_df.rating

df_eda
def show_dist(df, col):

    print('Descriptive stats for {}'.format(col))

    print('-'*(len(col)+22))

    print(df.groupby('rating')[col].describe())

    bins = np.arange(df[col].min(), df[col].max() + 1)

    g = sns.FacetGrid(df, col='rating', size=5, hue='rating', palette="PuBuGn_d")

    g = g.map(sns.distplot, col, kde=False, norm_hist=True, bins=bins)

    plt.show()
show_dist(df_eda, 'count_words')
class CleanText(BaseEstimator, TransformerMixin):

    def remove_mentions(self, input_text):

        return re.sub(r'@\w+', '', input_text)

    

    def remove_urls(self, input_text):

        return re.sub(r'http.?://[^\s]+[\s]?', '', input_text)

    

    def emoji_oneword(self, input_text):

        # By compressing the underscore, the emoji is kept as one word

        return input_text.replace('_','')

    

    def remove_punctuation(self, input_text):

        # Make translation table

        punct = string.punctuation

        trantab = str.maketrans(punct, len(punct)*' ')  # Every punctuation symbol will be replaced by a space

        return input_text.translate(trantab)



    def remove_digits(self, input_text):

        return re.sub('\d+', '', input_text)

    

    def to_lower(self, input_text):

        return input_text.lower()

    

    def remove_stopwords(self, input_text):

        stopwords_list = stopwords.words('english')

        # Some words which might indicate a certain sentiment are kept via a whitelist

        whitelist = ["n't", "not", "no"]

        words = input_text.split() 

        clean_words = [word for word in words if (word not in stopwords_list or word in whitelist) and len(word) > 1] 

        return " ".join(clean_words) 

    

    def stemming(self, input_text):

        porter = PorterStemmer()

        words = input_text.split() 

        stemmed_words = [porter.stem(word) for word in words]

        return " ".join(stemmed_words)

    

    def fit(self, X, y=None, **fit_params):

        return self

    

    def transform(self, X, **transform_params):

        clean_X = X.apply(self.remove_mentions).apply(self.remove_urls).apply(self.emoji_oneword).apply(self.remove_punctuation).apply(self.remove_digits).apply(self.to_lower).apply(self.remove_stopwords).apply(self.stemming)

        return clean_X
ct = CleanText()

sr_clean = ct.fit_transform(new_df.text)

sr_clean.sample(5)
df_model_clean = pd.DataFrame(sr_clean)

df_model_clean['rating'] = new_df.rating

df_model_clean
df_model_clean.columns.tolist()



X_train, X_test, y_train, y_test = train_test_split(df_model_clean.drop('rating', axis=1), df_model_clean.rating, test_size=0.3, random_state=50)



class ColumnExtractor(TransformerMixin, BaseEstimator):

    def __init__(self, cols):

        self.cols = cols



    def transform(self, X, **transform_params):

        return X[self.cols]



    def fit(self, X, y=None, **fit_params):

        return self


#create a new df model

df_model = df_eda

df_model['clean_text'] = pd.DataFrame(sr_clean)

df_model.columns.tolist()



textcountscols = ['count_capital_words','count_emojis','count_excl_quest_marks','count_hashtags','count_mentions','count_urls','count_words']



#textcountscols = ['text', 'rating']

    

features = FeatureUnion([('textcounts', ColumnExtractor(cols=textcountscols))

                         , ('pipe', Pipeline([('cleantext', ColumnExtractor(cols='clean_text'))

                                              , ('vect', CountVectorizer(max_df=0.5, min_df=1, ngram_range=(1,2)))]))]

                       , n_jobs=-1)



pipeline = Pipeline([

    ('features', features)

    , ('clf', LogisticRegression(C=1.0, penalty='l2'))

])



best_model = pipeline.fit(df_model.drop('rating', axis=1), df_model.rating)
new_reviews = pd.Series(["These sneakers feel great i had a great time using them i would buy them again in a heartbeat 👍"

                      ,"This soap sucks i dont like it at all it felt like fat i would not buy again"

                      ,"This detergent felt like it cleansed everything the best thing i've brought i like it a lot 😃"])



df_counts_pos = tc.transform(new_reviews)

df_clean_pos = ct.transform(new_reviews)

df_model_pos = df_counts_pos

df_model_pos['clean_text'] = df_clean_pos



best_model.predict(df_model_pos).tolist()