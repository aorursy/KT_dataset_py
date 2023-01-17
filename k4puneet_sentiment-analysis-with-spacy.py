!pip install pandarallel swifter
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are availaable in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
path = "/kaggle/input/aclimdb/aclImdb/"

positiveFiles = [x for x in os.listdir(path+"train/pos/") if x.endswith(".txt")]

negativeFiles = [x for x in os.listdir(path+"train/neg/") if x.endswith(".txt")]

testFiles = [x for x in os.listdir(path+"test/") if x.endswith(".txt")]
testFiles
positiveReviews, negativeReviews, testReviews = [], [], []

for pfile in positiveFiles:

    with open(path+"train/pos/"+pfile, encoding="latin1") as f:

        positiveReviews.append(f.read())

for nfile in negativeFiles:

    with open(path+"train/neg/"+nfile, encoding="latin1") as f:

        negativeReviews.append(f.read())

for tfile in testFiles:

    with open(path+"test/"+tfile, encoding="latin1") as f:

        testReviews.append(f.read())

        

reviews = pd.concat([

    pd.DataFrame({"review":positiveReviews, "label":1, "file":positiveFiles}),

    pd.DataFrame({"review":negativeReviews, "label":0, "file":negativeFiles}),

    pd.DataFrame({"review":testReviews, "label":-1, "file":testFiles})

], ignore_index=True).sample(frac=1, random_state=1)

reviews.head()
reviews.drop_duplicates(keep='first',inplace=True)

reviews.drop(['file'],1,inplace=True)
import spacy

import en_core_web_sm

from spacy.lang.en import English

from  spacy.lang.en.stop_words import STOP_WORDS

import string





import pandas as pd

from sklearn.feature_extraction import DictVectorizer

from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer,HashingVectorizer

from sklearn.base import TransformerMixin

from sklearn.pipeline import Pipeline



from bs4 import BeautifulSoup

import html as ihtml

import re

# custom transformer using spaCy

class RemoveHTMLTags(TransformerMixin):

    def transform(self, X, **transform_params):

        # cleaning text

        return X.apply(lambda text: clean_text(text))

        #return X.apply(lambda text: pd.Series({ "cleaned_review": clean_text(text) }),1 )

    

    def fit(self, X, y=None, **fit_params):

        return self

    

    def get_params(self, deep=True):

        return {}



# Basic function to clean the text

def clean_text(text):

    text = BeautifulSoup(ihtml.unescape(text)).text

    text = re.sub(r"http[s]?://\S+", "", text)

    text = re.sub(r"\s+", " ", text)

    text = re.sub(r"[^a-zA-Z ]"," ",text)

    return text
nlp = en_core_web_sm.load()

punctuations = string.punctuation

#nlp = spacy.load('en')



# To build a list of stop words for filtering

stopwords = list(STOP_WORDS)

parser = English()

#print(stopwords)

from pandarallel import pandarallel

pandarallel.initialize()

import swifter
class ExtractTextFeatures(TransformerMixin):

    def __init__(self):

        pass



    def fit(self, reviews, y=None):

        """ This is primarily used for NLP parsing """

        return self

    

    def get_params(self, **kwargs):

        return {}

    

    def pre_extract_text_features(self, X):

        """ To be used when features are pre-computed. """

        review_text = X['cleaned_review']

        word_count_length = len(X['tokens'])

        word_density = len(review_text) / word_count_length

        

        noun_count = len([x for x in X['pos'] if x == 'NOUN'])

        verb_count = len([x for x in X['pos'] if x == 'VERB'])

        adj_count = len([x for x in X['pos'] if x == 'ADJ'])

        

        mytokens = [ w if w != '-pron-' else i for i,w in zip(X['tokens'],X['lemma']) ]

    

        parsed_review = " ".join(mytokens)

        

        return pd.Series({'parsed_review': parsed_review, 

                          'word_count': word_count_length, 

                          'word_density': word_density, 

                          'noun_count': noun_count, 

                          'verb_count':  verb_count,

                          'adj_count' : adj_count

                         })

        

    def extract_text_features(self, X):

        doc = nlp(X,disable=['parser', 'ner'])

        

        word_count_length = len(doc)

        word_density = len(X)/word_count_length

        noun_count = len([x for x in doc if x.pos_ == 'NOUN'])

        verb_count = len([x for x in doc if x.pos_ == 'VERB'])

        adj_count = len([x for x in doc if x.pos_ == 'ADJ'])

        

        

        ## Creating a tokenized review text

        # lemmatizer

        mytokens = [ word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in doc ]

         # Removing stop words

        mytokens = [ word for word in mytokens if word not in stopwords and word not in punctuations ]

        parsed_review = " ".join(mytokens)

        

        return pd.Series({'parsed_review': parsed_review, 

                          'word_count': word_count_length, 

                          'word_density': word_density, 

                          'noun_count': noun_count, 

                          'verb_count':  verb_count

                         })

    

    

    def transform(self, X, y=None):

        """ Workhorse for the transformer """

        # assume X to be cleaned_review



        return X.apply(lambda text: self.extract_text_features(text), 1)
from tempfile import mkdtemp

from sklearn.preprocessing import FunctionTransformer

from sklearn.pipeline import make_pipeline, make_union

from sklearn.model_selection import train_test_split

from sklearn.compose import ColumnTransformer, make_column_transformer

cbow_processing_pipeline = make_pipeline(

        RemoveHTMLTags(),

        ExtractTextFeatures(),

        make_union(

            make_pipeline(

                FunctionTransformer(lambda x: x['parsed_review'],validate=False),

                CountVectorizer(ngram_range=(1,1))



            ),

            FunctionTransformer(lambda x: x.loc[:,'word_count'].values[:,np.newaxis],validate=False)

        ),



 

        #memory = cachedir,



)



cbow_processing_pipeline_pre = make_column_transformer(

    (cbow_processing_pipeline,'review')

)
tfidf_processing_pipeline = make_pipeline(

        RemoveHTMLTags(),

        ExtractTextFeatures(),

        make_union(

            

            make_pipeline(

                FunctionTransformer(lambda x: x['parsed_review'],validate=False),

                TfidfVectorizer(sublinear_tf=True, min_df=0.0025, max_df = 0.4,

                        ngram_range=(1, 3), 

                        stop_words=stopwords)



            ),

            FunctionTransformer(lambda x: x.loc[:,'word_count'].values[:,np.newaxis],validate=False)

        ),



 

        #memory = cachedir,



)



tfidf_processing_pipeline_pre = make_column_transformer(

    (tfidf_processing_pipeline,'review')

)
train = reviews[reviews.label.isin([0,1])]

test = reviews[reviews.label.isin([-1])].drop('label',1)



X_train, X_test, y_train, y_test = train_test_split(train.drop('label',1), train.label)
X_train.reset_index(inplace=True,drop=True)
X_train_preprocessed_tfidf = tfidf_processing_pipeline.fit_transform(X_train.review)

X_train_preprocessed_cbow = cbow_processing_pipeline.fit_transform(X_train.review)
X_train_preprocessed_tfidf
vocab = tfidf_processing_pipeline.named_steps['featureunion'].transformer_list[0][1].named_steps['tfidfvectorizer']

fn = tfidf_processing_pipeline.named_steps['featureunion'].transformer_list[0][1].named_steps['tfidfvectorizer'].get_feature_names()
tfidf = X_train_preprocessed_tfidf[:,:-1]
tfidf.shape, len(fn)
# Function to get top features in the row of tfidf vector



def top_tfidf_feats(row , features, top_n = 25):

    """ Get top n tfidf values in row and return them with their feature names."""

    topn_ids = np.argsort(row)[::-1][:top_n]

    top_feats = [(features[i], row[i]) for i in topn_ids]

    df = pd.DataFrame(top_feats)

    df.columns = ['feature','tfidf']

    return df



# function to call top_tfidf_feats



def top_feats_in_doc(Xtr, features, row_id, top_n=25):

    ''' Top tfidf features in specific document (matrix row) '''

    row = np.squeeze(Xtr[row_id].toarray())

    return top_tfidf_feats(row, features, top_n)



# calculate mean of tfidf values per group



def top_mean_feats(Xtr, features, grp_ids=None, min_tfidf=0.1, top_n=25):

    ''' Return the top n features that on average are most important amongst documents in rows

        indentified by indices in grp_ids. '''

    if grp_ids:

        D = Xtr[grp_ids].toarray()

    else:

        D = Xtr.toarray()



    D[D < min_tfidf] = 0

    tfidf_means = np.mean(D, axis=0)

    return top_tfidf_feats(tfidf_means, features, top_n)



# to call the above function by class label



def top_feats_by_class(Xtr, y, features, min_tfidf=0.1, top_n=25):

    ''' Return a list of dfs, where each df holds top_n features and their mean tfidf value

        calculated across documents with the same class label. '''

    dfs = []

    labels = np.unique(y)

    for label in labels:

        ids = np.where(y==label)

        feats_df = top_mean_feats(Xtr, features, ids, min_tfidf=min_tfidf, top_n=top_n)

        feats_df.label = label

        dfs.append(feats_df)

    return dfs
top_feats_in_doc(tfidf, fn, 4)
X_train.review[4]
dfs = top_feats_by_class(tfidf, y_train, fn)
import matplotlib.pyplot as plt



def plot_tfidf_classfeats_h(dfs):

    ''' Plot the data frames returned by the function plot_tfidf_classfeats(). '''

    fig = plt.figure(figsize=(12, 9), facecolor="w")

    x = np.arange(len(dfs[0]))

    for i, df in enumerate(dfs):

        ax = fig.add_subplot(1, len(dfs), i+1)

        ax.spines["top"].set_visible(False)

        ax.spines["right"].set_visible(False)

        ax.set_frame_on(False)

        ax.get_xaxis().tick_bottom()

        ax.get_yaxis().tick_left()

        ax.set_xlabel("Mean Tf-Idf Score", labelpad=16, fontsize=14)

        ax.set_title("label = " + str(df.label), fontsize=16)

        ax.ticklabel_format(axis='x', style='sci', scilimits=(-2,2))

        ax.barh(x, df.tfidf, align='center', color='#3F5D7D')

        ax.set_yticks(x)

        ax.set_ylim([-1, x[-1]+1])

        yticks = ax.set_yticklabels(df.feature)

        plt.subplots_adjust(bottom=0.09, right=0.97, left=0.15, top=0.95, wspace=0.52)

    plt.show()
plot_tfidf_classfeats_h(dfs)
vocab = cbow_processing_pipeline.named_steps['featureunion'].transformer_list[0][1].named_steps['countvectorizer']

fn = vocab.get_feature_names()
tfidf2 = X_train_preprocessed_cbow[:,:-1]



p1 = np.squeeze(np.asarray(tfidf2[np.where(y_train==1)].sum(0)))

p0 = np.squeeze(np.asarray(tfidf2[np.where(y_train==0)].sum(0)))



pr1 = (p1+1) / (p1.sum() + 1)

pr0 = (p0+1) / (p0.sum() + 1)
r = np.log(pr1/pr0); r


biggest = np.argpartition(r, -10)[-10:]

smallest = np.argpartition(r, 10)[:10]
itos = ['']*len(fn)



for k,v in vocab.vocabulary_.items():

    itos[v] = k



[itos[k] for k in biggest]

[itos[k] for k in smallest]


pd.value_counts(y_train).plot.bar()

# No class imbalance problem
from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from xgboost.sklearn import XGBClassifier



from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.metrics import accuracy_score, classification_report

from sklearn.pipeline import Pipeline

from sklearn.decomposition import TruncatedSVD



# Logistic Regression



lr_model = Pipeline([

    ("preprocessor", tfidf_processing_pipeline_pre),

    ("svd", TruncatedSVD(n_components=100, n_iter=7, random_state=42)),

    ("model", LogisticRegression(class_weight='balanced', solver='liblinear'))

])



# Decision Tree

dt_model = Pipeline([

    ("preprocessor", tfidf_processing_pipeline_pre),

        ("svd", TruncatedSVD(n_components=100, n_iter=7, random_state=42)),



    ("model", DecisionTreeClassifier(class_weight='balanced'))

])



# Random Forest

rf_model = Pipeline([

    ("preprocessor", tfidf_processing_pipeline_pre),

        ("svd", TruncatedSVD(n_components=100, n_iter=7, random_state=42)),



    ("model", RandomForestClassifier(class_weight='balanced', n_estimators=100, n_jobs=-1))

])



# XGBoost

xgb_model = Pipeline([

    ("preprocessor", tfidf_processing_pipeline_pre),

    ("svd", TruncatedSVD(n_components=100, n_iter=7, random_state=42)),



    ("model", XGBClassifier(scale_pos_weight=(1- y_train.mean()), n_jobs=-1))

])
gs = GridSearchCV(lr_model, {'model__C':[1, 1.3, 1.5]}, n_jobs=-1, cv=5, scoring='accuracy')

gs.fit(X_train, y_train)
print(gs.best_params_)

print(gs.best_score_)
lr_model.set_params(**gs.best_params_)
lr_model.fit(X_train, y_train)
y_pred = lr_model.predict(X_test)
accuracy_score(y_test, y_pred)
print(classification_report(y_test, y_pred))
import eli5

eli5.show_weights(lr_model.named_steps['model'])
gs = GridSearchCV(dt_model, {"model__max_Depth": [3,5,7],

                            "model__min_samples_split": [2,5]},

                 n_jobs=-1, cv=5, scoring='accuracy')



gs.fit(X_train, y_train)
print(gs.best_params_)

print(gs.best_score_)
dt_model.set_params(**gs.best_params_)
dt_model.fit(X_train, y_train)

y_pred = dt_model.predict(X_test)
accuracy_score(y_test, y_pred)
print(classification_report(y_test, y_pred))
gs = GridSearchCV(rf_model, {"model__max_depth": [10, 15],

                            "mdel__min_samples_split": [5,10]},

                 n_jobs=-1, cv=5, scoring='accuracy')

gs.fit(X_train, y_train)
print(gs.best_params_)

print(gs.best_score_)
rf_model.set_params(**gs.best_params_)
rf_model.fit(X_train, y_train)

y_pred = rf_model.predict(X_test)
accuracy_score(y_test, y_pred)
print(classification_report(y_test, y_pred))
gs = GridSearchCV(xgb_model, {'model__max_depth': [5,10],

                             "model__min_child_weight": [5,10],

                             "model__n_estimators":[25]},

                 n_jobs=-1, cv=5, scoring='accuracy')



gs.fit(X_train, y_train)
print(gs.best_params_)

print(gs.best_score_)

xgb_model.set_params(**gs.best_params_)

xgb_model.fit(X_train, y_train)
y_pred = xgb_model.predict(X_test)
accuracy_score(y_test, y_pred)
print(classification_report(y_test, y_pred))
X_train_preprocessed_tfidf = tfidf_processing_pipeline.fit_transform(X_train)

X_train_preprocessed_cbow = cbow_processing_pipeline.fit_transform(X_train)