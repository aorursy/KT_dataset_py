
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from pandas_profiling import ProfileReport
import plotly.express as px


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.

# import joypy
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns; sns.set()
%matplotlib inline

from sklearn.datasets import load_breast_cancer

# garbage
import gc; gc.enable()

# warnings
import warnings
warnings.filterwarnings("ignore")

# modeling
from sklearn.naive_bayes import GaussianNB
# from sklego.mixture import GMMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, roc_curve
from sklearn.metrics import precision_score, recall_score, accuracy_score, make_scorer
from sklearn.model_selection import GridSearchCV
# from sklego.meta import Thresholder
from sklearn.pipeline import make_pipeline
instruments = pd.read_json('/kaggle/input/amazon-music-reviews/Musical_Instruments_5.json', lines = True)
reviews = pd.read_csv('/kaggle/input/amazon-music-reviews/Musical_instruments_reviews.csv')
assert reviews.shape == instruments.shape
train_profile = ProfileReport(reviews, title='Pandas Profiling Report', html={'style':{'full_width':True}})
train_profile
reviews
reviews.overall.value_counts(normalize=True)

fig = px.histogram(reviews, x="overall")
fig.show()
reviews['log_overall'] = np.log1p(reviews['overall']) 
fig = px.histogram(reviews, x="log_overall")
fig.show()
#check amount of reviewers
reviews.reviewerID.nunique()
reviews.reviewerID.value_counts(ascending=False)
reviews[reviews.reviewerID == 'ADH0O8UVJOT10']
reviews.groupby('reviewerID')['overall'].agg(['mean','count']).sort_values(by='mean')
reviews[reviews.reviewerID == 'A1B3CNORXB1USI']
reviews.info()
target = reviews['overall']
from nltk.corpus import wordnet
import string
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.tokenize import WhitespaceTokenizer
from nltk.stem import WordNetLemmatizer
def get_wordnet_pos(pos_tag):
    if pos_tag.startswith('J'):
        return wordnet.ADJ
    elif pos_tag.startswith('V'):
        return wordnet.VERB
    elif pos_tag.startswith('N'):
        return wordnet.NOUN
    elif pos_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN
def clean_text(text):

    # lower text
    text = text.lower()
    # tokenize text and remove puncutation
    text = [word.strip(string.punctuation) for word in text.split(" ")]
    # remove words that contain numbers
    text = [word for word in text if not any(c.isdigit() for c in word)]

    # remove stop words
    stop = stopwords.words('english')
    text = [x for x in text if x not in stop]
    # remove empty tokens
    text = [t for t in text if len(t) > 0]
    # pos tag text
    pos_tags = pos_tag(text)
    # lemmatize text
    text = [WordNetLemmatizer().lemmatize(t[0], get_wordnet_pos(t[1])) for t in pos_tags]
    # remove words with only one letter
    text = [t for t in text if len(t) > 1]
    # join all
    text = " ".join(text)
    return(text)
reviews["reviewText"] = reviews["reviewText"].astype(str)
reviews['clean_review'] = reviews["reviewText"].apply(lambda x: clean_text(x))
reviews
from nltk.sentiment.vader import SentimentIntensityAnalyzer

sid = SentimentIntensityAnalyzer()
reviews["sentiments"] = reviews['clean_review'].apply(lambda x: sid.polarity_scores(x))
reviews_df = pd.concat([reviews.drop(['sentiments'], axis=1), reviews['sentiments'].apply(pd.Series)], axis=1)
reviews_df

# add number of characters column
reviews["nb_chars"] = reviews["reviewText"].apply(lambda x: len(x))

# add number of words column
reviews["nb_words"] = reviews["reviewText"].apply(lambda x: len(x.split(" ")))
!pip install gensim


# create doc2vec vector columns
from gensim.test.utils import common_texts
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(reviews_df["clean_review"].apply(lambda x: x.split(" ")))]

# train a Doc2Vec model with our text data
model = Doc2Vec(documents, vector_size=5, window=2, min_count=1, workers=4)

# transform each document into a vector data
doc2vec_df = reviews_df["clean_review"].apply(lambda x: model.infer_vector(x.split(" "))).apply(pd.Series)
doc2vec_df.columns = ["doc2vec_vector_" + str(x) for x in doc2vec_df.columns]
reviews_df = pd.concat([reviews_df, doc2vec_df], axis=1)
t_profile = ProfileReport(reviews_df, title='Pandas Profiling Report', html={'style':{'full_width':True}})
t_profile

from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(min_df = 10)
tfidf_result = tfidf.fit_transform(reviews_df["clean_review"]).toarray()
tfidf_df = pd.DataFrame(tfidf_result, columns = tfidf.get_feature_names())
tfidf_df.columns = ["word_" + str(x) for x in tfidf_df.columns]
tfidf_df.index = reviews_df.index
reviews_df = pd.concat([reviews_df, tfidf_df], axis=1)
reviews_df.columns[:10]

ignore_cols = reviews_df.columns[:10].tolist()
used_cols = [c for c in reviews_df.columns.tolist() if c not in ignore_cols]
from sklearn.model_selection import cross_val_score, train_test_split
from xgboost import XGBRFRegressor
from sklearn.linear_model import LinearRegression, MultiTaskElasticNet
import sklearn
xgb = XGBRFRegressor()
lr = LinearRegression()
sorted(sklearn.metrics.SCORERS.keys())
scores_xgb = cross_val_score(xgb,reviews_df[used_cols], reviews_df['overall'], cv=5, scoring='neg_median_absolute_error')
scores_xgb
scores_lr = cross_val_score(lr,reviews_df[used_cols], reviews_df['overall'], cv=5, scoring= 'neg_median_absolute_error')
scores_lr
X_train, X_test, y_train, y_test = train_test_split(reviews_df[used_cols], reviews_df['overall'])
xgb.fit(X_train,y_train)
xgb_preds = xgb.predict(X_test)
print(xgb_preds.mean(),
      xgb_preds.std(),
         xgb_preds.min(),
     xgb_preds.max())
     
print(y_test.mean(),
        y_test.std())
