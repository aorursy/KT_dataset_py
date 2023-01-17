import numpy as np

import pandas as pd

from sklearn import feature_extraction, linear_model, model_selection, preprocessing
train_df = pd.read_csv("/kaggle/input/nlp-getting-started/train.csv", dtype={'id':np.int16,'target':np.int8})

test_df = pd.read_csv("/kaggle/input/nlp-getting-started/test.csv", dtype={'id':np.int16})
train_df.head(1)
train_df[train_df["target"]==0]["text"].values[3]

#train_df[train_df["target"]==0]["text"].head(1)
train_df[train_df["target"] == 1]["text"].head(2)
print(train_df.shape)

print(test_df.shape)
print(f'Number of unique values in keyword = {train_df["keyword"].nunique()} (Training) - {test_df["keyword"].nunique()} (Test)')

print(f'Number of unique values in location = {train_df["location"].nunique()} (Training) - {test_df["location"].nunique()} (Test)')
#coverting everything into lowercase for stability

train_df['text'] =  train_df['text'].apply(lambda x: x.lower())

test_df['text'] =  test_df['text'].apply(lambda x: x.lower())



#tokenizing first

#I am using gensim here as it will remove all the punctuation automatically

from gensim.utils import tokenize

train_df['text'] =  train_df['text'].apply(lambda x: list(tokenize(x)))

test_df['text'] =  test_df['text'].apply(lambda x: list(tokenize(x)))



#If you want to take into account hash(#) or other useful symbol, it's better to use nltk

#from nltk.tokenize import word_tokenize 

#train_df['text'] =  train_df['text'].apply(lambda x: list(word_tokenize(x)))

#test_df['text'] =  test_df['text'].apply(lambda x: list(word_tokenize(x)))
# I am using gensim here as gensim have highest number of stopwords. Other libraries are NLTK and Spacy.

import gensim.parsing.preprocessing

stopwords =  gensim.parsing.preprocessing.STOPWORDS



#I am not creating here new column. If you wish, you can create new column when you want to test the both results.

train_df['text'] = train_df['text'].apply(lambda x: [item for item in x if item not in stopwords])

test_df['text'] = test_df['text'].apply(lambda x: [item for item in x if item not in stopwords])
# Stemming

from nltk.stem import SnowballStemmer

stemmer = SnowballStemmer('english')

train_df['text'] = train_df['text'].apply(lambda x: [stemmer.stem(item) for item in x])

test_df['text'] = test_df['text'].apply(lambda x: [stemmer.stem(item) for item in x])



# Lemmetization

from nltk.stem import WordNetLemmatizer

lemma = WordNetLemmatizer()

#at the end join every word

train_df['text'] = train_df['text'].apply(lambda x: lemma.lemmatize(" ".join(x)))

test_df['text'] = test_df['text'].apply(lambda x: lemma.lemmatize(" ".join(x)))

count_vectorizer = feature_extraction.text.CountVectorizer()

train_vectors = count_vectorizer.fit_transform(train_df["text"])

test_vectors = count_vectorizer.transform(test_df["text"])
classifier = linear_model.RidgeClassifier()
score = model_selection.cross_val_score(classifier,train_vectors, train_df['target'],cv=3,scoring='f1')

score
classifier.fit(train_vectors, train_df['target'])
sample_submission = pd.read_csv("/kaggle/input/nlp-getting-started/sample_submission.csv")
sample_submission.head()
sample_submission["target"] = classifier.predict(test_vectors)
sample_submission.head()
sample_submission.to_csv("submission.csv", index=False)
vectorizer = feature_extraction.text.TfidfVectorizer(use_idf=True, stop_words = 'english')

train_tfidf_vectors = vectorizer.fit_transform(train_df["text"])

test_tfidf_vectors = vectorizer.transform(test_df["text"])
score = model_selection.cross_val_score(classifier,train_tfidf_vectors, train_df['target'],cv=3,scoring='f1')

score
classifier.fit(train_tfidf_vectors, train_df['target'])
sample_submission["target"] = classifier.predict(test_tfidf_vectors)
sample_submission.head()
sample_submission.to_csv("submission.csv", index=False)
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier()
score = model_selection.cross_val_score(rf,train_tfidf_vectors, train_df['target'],cv=3,scoring='f1')

score
###I am not writing my submission file over here as i am not getting high accuracy



#rf.fit(train_tfidf_vectors, train_df['target'])

#sample_submission["target"] = rf.predict(test_tfidf_vectors)

#sample_submission.to_csv("submission.csv", index=False)
from sklearn.ensemble import GradientBoostingClassifier

#gb = GradientBoostingClassifier(random_state=0,learning_rate=0.01)
"""

print(dir(GradientBoostingClassifier))

print(gb)

"""
"""

import multiprocessing

n_jobs = multiprocessing.cpu_count()-1

print(n_jobs)

"""
"""

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import accuracy_score, precision_score, recall_score, make_scorer





scoring = {'accuracy': make_scorer(accuracy_score),

           'precision': make_scorer(precision_score),'recall':make_scorer(recall_score)}



parameters = {

    "loss":["deviance"],

    "learning_rate": [0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2],

    "min_samples_split": np.linspace(0.1, 0.5, 12),

    "min_samples_leaf": np.linspace(0.1, 0.5, 12),

    "max_depth":[3,5,8],

    "max_features":["log2","sqrt"],

    "criterion": ["friedman_mse",  "mae"],

    "subsample":[0.5, 0.618, 0.8, 0.85, 0.9, 0.95, 1.0],

    "n_estimators":[10]

    }



grid_search = GridSearchCV(GradientBoostingClassifier(), parameters,scoring=scoring,refit=False,cv=2, n_jobs=n_jobs)



grid_search.fit(train_tfidf_vectors, train_df['target'])

df=pd.DataFrame.from_dict(grid_search.cv_results_)

print(df)

#df[['split0_test_accuracy','split1_test_accuracy','split0_test_precision','split1_test_precision','split0_test_recall','split1_test_recall']]

print(grid_search.best_score_)

print(grid_search.best_params_)



"""



####This takes tooooooo much time. So i am just ignoring it right now.
gb = GradientBoostingClassifier(random_state=0,learning_rate=0.01)
score = model_selection.cross_val_score(gb,train_tfidf_vectors, train_df['target'],cv=3,scoring='f1')

score
gb.fit(train_tfidf_vectors, train_df['target'])

sample_submission["target"] = gb.predict(test_tfidf_vectors)

sample_submission.to_csv("submission.csv", index=False)
from xgboost import XGBClassifier

xgb = XGBClassifier(random_state=0,learning_rate=0.0001,max_depth=6,min_child_weight =1,n_jobs=5,n_estimators=100)
xgb
score = model_selection.cross_val_score(xgb,train_tfidf_vectors, train_df['target'],cv=3,scoring='f1')

score
xgb.fit(train_tfidf_vectors, train_df['target'])

sample_submission["target"] = gb.predict(test_tfidf_vectors)

sample_submission.to_csv("submission.csv", index=False)