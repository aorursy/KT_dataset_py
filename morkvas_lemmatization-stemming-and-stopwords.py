import os

import numpy as np

import pandas as pd

import eli5

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn import metrics

import seaborn as sns

from scipy.sparse import csr_matrix, hstack

from datetime import datetime

import string

from nltk.stem import PorterStemmer, WordNetLemmatizer
# Load the data. There is a test dataset in the folder, but we cannot use it because we have no answers. Therefore we create train and test sets from the training data

sarcasm_df = pd.read_csv("../input/sarcasm/train-balanced-sarcasm.csv")
sarcasm_df.head()
# Here we inspect data for missing values

sarcasm_df.info()
# We can see, that comment row has missing values: comment 1010773 non-null and it should be 1010825. We will delete these rows completely.

# We also drop columns 'ups', 'downs', 'date' and convert string 'created_utc' to datetime format

sarcasm_df.dropna(subset=['comment'], inplace=True)

sarcasm_df.drop(['ups', 'downs', 'date'], axis=1)

sarcasm_df['created_utc'] = sarcasm_df['created_utc'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
# Now we compare the number of instances for each class (1 - sarcasm, 0 - not). We can see, that the dataset is balanced and classes have almost the same size

sarcasm_df['label'].value_counts()
# Here we split our data for the train and test sets

train_df, test_df, train_y, test_y = train_test_split(sarcasm_df, sarcasm_df['label'], test_size=0.33, random_state=17)
eda_data = train_df[['comment', 'author', 'subreddit','created_utc', 'score','parent_comment']].copy()

eda_data['label'] = train_y
eda_data.head()
filtered = eda_data.groupby(['subreddit']).filter(lambda x: x['comment'].count()>1000)

filtered.groupby(['subreddit']).agg({

    'comment':'count',

    'label': 'mean'}).sort_values(by='label', ascending=False).iloc[:10]
filtered_authors = eda_data.groupby(['author']).filter(lambda x: x['comment'].count()>20)

filtered_authors.groupby(['author']).agg({

    'comment':'count',

    'label': 'mean'}).sort_values(by='label', ascending=False).iloc[:10]
# Here we add the new field -  comment_len

eda_data['comment_len'] = eda_data['comment'].apply(len)
def percentile_print(data, feature, percentile_list = [25, 50, 75]):

    for percentile in percentile_list:

        print ("Percentile",percentile,

               "Sarcasm", np.percentile(data[data['label']==1][feature], percentile),

               "Not", np.percentile(data[data['label']==0][feature], percentile))
# So we see, that there no difference in comment length for both classes. That'a a bit dissapointing, but we will move forward

percentile_print(eda_data, 'comment_len')
# But unfortunatelly, the situation is the same for scores too: no visible difference.

percentile_print(eda_data, 'score')
eda_data['weekend'] = eda_data['created_utc'].apply(lambda x: x.dayofweek==1 or x.dayofweek==6).astype(int)

eda_data['day']= eda_data['created_utc'].apply(lambda x: x.hour>7 and x.hour<20).astype(int)
# So what have we here? Working days don't make users more sarcastic.

sns.countplot(x='weekend', hue='label', data=eda_data )
# And finally here we can see the difference! Day - that's the time for the sarcasm =)

sns.countplot(x='day', hue='label', data=eda_data )
vectorizer_1 = CountVectorizer(stop_words='english', ngram_range=(1, 1))

vectorizer_2 = CountVectorizer(stop_words='english', ngram_range=(2, 2))
def freq_words(vectorizer, data):

    X = vectorizer.fit_transform(data)

    freqs = zip(vectorizer.get_feature_names(), np.asarray(X.sum(axis=0)).ravel())

    return sorted(freqs, key = lambda x: x[1], reverse=True)[:10]
# First column - sarcastic comments, second - not

l = [freq_words(vectorizer_1, eda_data[eda_data['label']==1]['comment']),

     freq_words(vectorizer_1, eda_data[eda_data['label']==0]['comment'])]

list(map(list, zip(*l)))
# First column - sarcastic comments, second - not

l = [freq_words(vectorizer_2, eda_data[eda_data['label']==1]['comment']),

     freq_words(vectorizer_2, eda_data[eda_data['label']==0]['comment'])]

list(map(list, zip(*l)))
# Again a small function for finding the intersection. This one does the following:

# 1) set characters in the string to lowercase, delete punctuation and split for the words

# 2) the same for the parent comment

# 3) find words in the comment, that are also in the parent comment

# 4) returns the rate of intersection length to the length of all words in the comment 



def find_intersection(comment, parent):

    comment_words = [x.strip(string.punctuation) for x in comment.lower().split()]

    parent_words = [x.strip(string.punctuation) for x in parent.lower().split()]

    intersection_words = [x for x in comment_words if x in parent_words]

    return len(intersection_words)/len(comment_words)
# Now we add this intersection feature to our dataframe and will look at it

eda_data['intersection'] = [find_intersection(x,y) for x,y in zip(eda_data['comment'], eda_data['parent_comment'])]
sns.boxplot(x = eda_data[eda_data['label']==1]['intersection'])
sns.boxplot(x = eda_data[eda_data['label']==0]['intersection'])
percentile_print(eda_data, 'intersection')
log_reg = LogisticRegression(random_state=17, solver='lbfgs')
# Function for printing metrics for our predicted result

def print_report(model, x_test, y_test):

    y_pred = model.predict(x_test)

    report = metrics.classification_report(y_test, y_pred)

    print(report)

    print("accuracy: {:0.3f}".format(metrics.accuracy_score(y_test, y_pred)))
# This function helps us not to repeat the same lines of code many times. Here we:

# 1) make pipeline

# 2) train it

# 3) print metrics

# 4) return our trained regression and feature_names, so we will be able to look at the weights

def model_cycle(vectorizer, train_x=train_df['comment'], test_x=test_df['comment']):

    train_vect = vectorizer.fit_transform(train_x)

    test_vect = vectorizer.transform(test_x)

    log_reg.fit(train_vect,train_y)

    print_report(log_reg, test_vect , test_y)

    return (log_reg, vectorizer.get_feature_names())
# This code takes some time to run, be patient

(model, features) = model_cycle(CountVectorizer(ngram_range=(1, 3), max_features=100000))
eli5.show_weights(model,

                  feature_names=features,

                  target_names = ['0','1'],

                  )
# This code takes some time to run, be patient

(model, features) = model_cycle(TfidfVectorizer(ngram_range=(1, 3), max_features=100000))
eli5.show_weights(model,

                  feature_names=features,

                  target_names = ['0','1'],

                  )
(model, features) = model_cycle(CountVectorizer(ngram_range=(1, 3), stop_words='english', max_features=100000))
eli5.show_weights(model,

                  feature_names=features,

                  target_names = ['0','1'],

                  )
(model, features) = model_cycle(TfidfVectorizer(ngram_range=(1, 3), stop_words='english', max_features=100000))
eli5.show_weights(model,

                  feature_names=features,

                  target_names = ['0','1'],

                  )
stemmer = PorterStemmer()

class StemmedCountVectorizer(CountVectorizer):

    def build_analyzer(self):

        analyzer = super(StemmedCountVectorizer, self).build_analyzer()

        return lambda doc: ([stemmer.stem(w) for w in analyzer(doc)])



vectorizer_s = StemmedCountVectorizer(analyzer="word", ngram_range=(1, 3), max_features=100000)
# This code runs realy for a long time. If you want to re-run it be patient.

(model, features) = model_cycle(vectorizer_s)
eli5.show_weights(model,

                  feature_names=features,

                  target_names = ['0','1'],

                  )
class StemmedTfidfVectorizer(TfidfVectorizer):

    def build_analyzer(self):

        analyzer = super(StemmedTfidfVectorizer, self).build_analyzer()

        return lambda doc: ([stemmer.stem(w) for w in analyzer(doc)])
# This code also takes a lot of time. You can make a tea and talk with friends a bit

(model, features) = model_cycle(StemmedTfidfVectorizer(analyzer="word", ngram_range=(1, 3), max_features=100000))
eli5.show_weights(model,

                  feature_names=features,

                  target_names = ['0','1'],

                  )
from nltk import word_tokenize

from nltk.corpus import wordnet 



class LemmaTokenizer(object):

    def __init__(self):

        self.wnl = WordNetLemmatizer()

    def __call__(self, articles):

        return [self.wnl.lemmatize(t,wordnet.VERB) for t in word_tokenize(articles)]



vectorizer = CountVectorizer(tokenizer=LemmaTokenizer(),

                             strip_accents = 'unicode',

                             lowercase = True,

                             ngram_range=(1, 3), max_features=100000)



#stripping punctuation here

train_stripped_comment = train_df['comment'].str.replace('[^\w\s]', '')

test_stripped_comment = test_df['comment'].str.replace('[^\w\s]', '')
(model, features) = model_cycle(vectorizer,train_stripped_comment, test_stripped_comment)
eli5.show_weights(model,

                  feature_names=features,

                  target_names = ['0','1'],

                  )
tf_vectorizer = TfidfVectorizer(tokenizer=LemmaTokenizer(),

                                strip_accents = 'unicode',

                                lowercase = True,

                                ngram_range=(1, 3), max_features=100000)
(model, features) = model_cycle(tf_vectorizer,train_stripped_comment, test_stripped_comment)
eli5.show_weights(model,

                  feature_names=features,

                  target_names = ['0','1'],

                  )
# Define vectorizer and convert our text features

train_comment = tf_vectorizer.fit_transform(train_stripped_comment)

test_comment = tf_vectorizer.transform(test_stripped_comment)
# Subreddit feature should be codded with OneHotEncoder, after it every unique subreddit value will be the separate feature

enc_sub = OneHotEncoder(handle_unknown='ignore')

train_subreddit = enc_sub.fit_transform(train_df['subreddit'].values.reshape(-1,1))

test_subreddit = enc_sub.transform(test_df['subreddit'].values.reshape(-1,1))
# The same for the author field

enc_aut = OneHotEncoder(handle_unknown='ignore')

train_author = enc_aut.fit_transform(train_df['author'].values.reshape(-1,1))

test_author = enc_aut.transform(test_df['author'].values.reshape(-1,1))
# We will scale our real-valued features

scaler = StandardScaler()

train_scores = scaler.fit_transform(train_df['score'].values.reshape(-1,1))

test_scores = scaler.transform(test_df['score'].values.reshape(-1,1))

train_len = scaler.fit_transform((train_df['comment'].apply(len)).values.reshape(-1,1))

test_len = scaler.transform((test_df['comment'].apply(len)).values.reshape(-1,1))
# And finally here we append day, weekend and intersection features



train_df['day'] = train_df['created_utc'].apply(lambda x: x.hour>7 and x.hour<20).astype(int)

test_df['day'] = test_df['created_utc'].apply(lambda x: x.hour>7 and x.hour<20).astype(int)



train_df['intersection'] = [find_intersection(x,y) for x,y in zip(train_df['comment'], train_df['parent_comment'])]

test_df['intersection'] = [find_intersection(x,y) for x,y in zip(test_df['comment'], test_df['parent_comment'])]



train_df['weekend'] = train_df['created_utc'].apply(lambda x: x.dayofweek==1 or x.dayofweek==6).astype(int)

test_df['weekend'] = test_df['created_utc'].apply(lambda x: x.dayofweek==1 or x.dayofweek==6).astype(int)
# Here we combine all our features in the big sparse matrix

train_sparse = hstack([train_comment, train_subreddit, train_author, train_scores, train_len,

                       train_df['day'].values.reshape(-1,1), train_df['intersection'].values.reshape(-1,1),

                       train_df['weekend'].values.reshape(-1,1)]).tocsr()

test_sparse = hstack([test_comment, test_subreddit, test_author, test_scores, test_len,

                      test_df['day'].values.reshape(-1,1), test_df['intersection'].values.reshape(-1,1),

                      test_df['weekend'].values.reshape(-1,1)]).tocsr()
# Fit the model and print report

log_reg.fit(train_sparse, train_y)

print_report(log_reg, test_sparse, test_y)
"""This is the commented part with parameters tuning

In our LogisticRegression model we have only one parameter C

and we need to make the pipeline so the train and validation data will not blend.

This code runs a long time, be ready.

"""

"""

from sklearn.pipeline import make_pipeline

from sklearn.model_selection import GridSearchCV



pipe_logit = make_pipeline(tf_vectorizer, log_reg)

param_grid_logit = {'logisticregression__C': np.logspace(-3, 1, 5)}



grid_logit = GridSearchCV(pipe_logit, 

                          param_grid_logit, 

                          return_train_score=True, 

                          cv=3, n_jobs=-1)



grid_logit.fit(train_stripped_comment, train_y)



grid_logit.best_params_, grid_logit.best_score_



grid_logit.score(test_stripped_comment,test_y)

"""