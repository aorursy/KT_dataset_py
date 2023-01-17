import numpy as np

import pandas as pd

import os

import copy

import string

import nltk

from nltk.corpus import stopwords

from nltk.tokenize import word_tokenize

from nltk.stem.wordnet import WordNetLemmatizer

from sklearn.preprocessing import LabelEncoder

from sklearn.feature_extraction.text import TfidfVectorizer

import matplotlib.pyplot as plt

import seaborn as sns

wnl = WordNetLemmatizer()

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

pd.set_option("display.max_colwidth", 80)

import modeling_functions as mf

import nlp_preprocessing_functions as npf

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.linear_model import RidgeClassifier

from sklearn.tree import DecisionTreeClassifier
train = pd.read_csv("/kaggle/input/nlp-getting-started/train.csv")

print('Train Set Shape = {}'.format(train.shape))

train.head()
test = pd.read_csv("/kaggle/input/nlp-getting-started/test.csv")

print('Test Set Shape = {}'.format(test.shape))

test.head()
submission = pd.read_csv("/kaggle/input/nlp-getting-started/sample_submission.csv")

submission.head()
missing_cols = ['keyword', 'location']



fig, axes = plt.subplots(ncols=2, figsize=(20, 5))



train_sums = train[missing_cols].isnull().sum()

test_sums = test[missing_cols].isnull().sum()

sns.barplot(x=train_sums.index, y=train_sums.values, ax=axes[0])

sns.barplot(x=test_sums.index, y=test_sums.values, ax=axes[1])



axes[0].set_ylabel('Missing Value Count', size=15, labelpad=20)

axes[0].set_title('Train Set', fontsize=13)

axes[1].set_title('Test Set', fontsize=13)

axes[0].tick_params(axis='x', labelsize=12)

axes[0].tick_params(axis='y', labelsize=12)

axes[1].tick_params(axis='x', labelsize=12)

axes[1].tick_params(axis='y', labelsize=12)



plt.show()



train_dropna, test_dropna = copy.copy(train), copy.copy(test)

for df in [train_dropna, test_dropna]:

    for col in ['keyword', 'location']:

        df[col] = df[col].fillna(f'no_{col}')
print(f'Unique Values in location column(train set) : {train_dropna["location"].nunique()} / {len(train_dropna["location"])}')

print(f'Unique Values in location column(test set) : {test_dropna["location"].nunique()} / {len(test_dropna["location"])}')
print(f'Unique Values in keyword column(train set) : {train_dropna["keyword"].nunique()} / {len(train_dropna["keyword"])}')

print(f'Unique Values in keyword column(test set) : {test_dropna["keyword"].nunique()} / {len(test_dropna["keyword"])}')
train_preprocessed, test_preprocessed = copy.copy(train_dropna), copy.copy(test_dropna)

train_preprocessed.drop(['location'], axis=1, inplace=True)

le = LabelEncoder()

le.fit(train_preprocessed['keyword'])

train_preprocessed['keyword'] = le.transform(train_preprocessed['keyword'])

train_preprocessed.head()
test_preprocessed.drop(['location'], axis=1, inplace=True)

test_preprocessed['keyword'] = le.transform(test_preprocessed['keyword'])

test_preprocessed.head()
train_preprocessed.drop(['id'], axis=1, inplace=True)

train_preprocessed.head()
test_preprocessed.drop(['id'], axis=1, inplace=True)

test_preprocessed.head()
sentences_raw_train = train_preprocessed['text']

sentences_raw_test = test_preprocessed['text']

sentences_preprocessed_train = []

sentences_preprocessed_test = []

for sentence in sentences_raw_train:

    lemmas = npf.tokenizer(sentence)

    sentence_without_stop_words = npf.remove_stop_words(lemmas, st_list=['amp','ca','ha','http http','new','rt','wa'])

    sentences_preprocessed_train.append(sentence_without_stop_words)

sentences_preprocessed_train = [" ".join(doc) for doc in sentences_preprocessed_train]

for sentence in sentences_raw_test:

    lemmas = npf.tokenizer(sentence)

    sentence_without_stop_words = npf.remove_stop_words(lemmas, st_list=['amp','ca','ha','http http','new','rt','wa'])

    sentences_preprocessed_test.append(sentence_without_stop_words)

sentences_preprocessed_test = [" ".join(doc) for doc in sentences_preprocessed_test]

sentences_tfidf = copy.copy(sentences_preprocessed_train)

sentences_tfidf.extend(sentences_preprocessed_test)
tfidf_train, tfidf_test = npf.tfidf_features(docs_tfidf=sentences_tfidf, docs_train=sentences_preprocessed_train, docs_test=sentences_preprocessed_test, _max_features=1000)

tfidf_train.head()
tfidf_test.head()
train_preprocessed = pd.concat([train_preprocessed, tfidf_train], axis=1)

train_preprocessed.drop(['text'], axis=1, inplace=True)

train_preprocessed.head()
test_preprocessed = pd.concat([test_preprocessed, tfidf_test], axis=1)

test_preprocessed.drop(['text'], axis=1, inplace=True)

test_preprocessed.head()
meta_feature_train = pd.DataFrame(columns=["word_count", "unique_word_count", "mean_word_count", "punctuation_count", "news_word_count", "disaster_word_count"])

meta_feature_test = pd.DataFrame(columns=["word_count", "unique_word_count", "mean_word_count", "punctuation_count", "news_word_count", "disaster_word_count"])



tokenized_train =  sentences_raw_train.apply(lambda x: npf.tokenizer(x))

tokenized_test =  sentences_raw_test.apply(lambda x: npf.tokenizer(x))



#word_count

meta_feature_train["word_count"] = tokenized_train.apply(lambda x: len(x))

meta_feature_test["word_count"] = tokenized_test.apply(lambda x: len(x))



#unique_word_count

meta_feature_train["unique_word_count"] = tokenized_train.apply(lambda x: len(set(x)))

meta_feature_test["unique_word_count"] = tokenized_test.apply(lambda x: len(set(x)))



#mean_word_count

meta_feature_train["mean_word_count"] = tokenized_train.apply(lambda x: np.mean([len(i) for i in x]))

meta_feature_test["mean_word_count"] = tokenized_test.apply(lambda x: np.mean([len(i) for i in x]))



#punctuation_count

meta_feature_train["punctuation_count"] = sentences_raw_train.apply(lambda x: len([c for c in x if c in string.punctuation]))

meta_feature_test["punctuation_count"] = sentences_raw_test.apply(lambda x: len([c for c in x if c in string.punctuation]))



#news_word_count

news_related_word = ["news", "report","pm", "am", "utc", "breaking", "bbc", "abc", "fox", "gov", "government", "whitehouse", "huff", "journal", "cbc", "cbs", "official", "officer", "cnn", "yorker", "yahoo", "tv", "radio"]

meta_feature_train["news_word_count"] = tokenized_train.apply(lambda x: len([w for w in x if w in news_related_word]))

meta_feature_test["news_word_count"] = tokenized_test.apply(lambda x: len([w for w in x if w in news_related_word]))



#disaster_word_count

disaster_related_word = ["disaster", "accudent", "kill", "killed", "killing", "died", "earthquake", "death", "bomb", "bombed", "bombing", "flood", "fire", "wildfire", "burn", "burning", "crash", "victims", "war", "weapons", "military", "force", "forces", "survive", "survived", "blood"]

meta_feature_train["disaster_word_count"] = tokenized_train.apply(lambda x: len([w for w in x if w in disaster_related_word]))

meta_feature_test["disaster_word_count"] = tokenized_test.apply(lambda x: len([w for w in x if w in disaster_related_word]))



train_preprocessed = pd.concat([train_preprocessed, meta_feature_train], axis=1)

train_preprocessed.head()
test_preprocessed = pd.concat([test_preprocessed, meta_feature_test], axis=1)

test_preprocessed.head()
y_train = train_preprocessed['target']

x_train = train_preprocessed.drop(['target'], axis=1)

x_test = test_preprocessed
target_model_set = {}



param_list_rf_1 = {

    "n_estimators" : ["int",[170,200]],

    "max_depth" : ["int",[6,8]],

    "random_state" : 1234

}

model_rf_1 = RandomForestClassifier()

target_model_set["random_forest_1"] = [param_list_rf_1, model_rf_1]



param_list_knn_1 = {

    "n_neighbors" : ["int",[5,20]],

    "weights" : "distance",

    "p" : ["int",[1,2]],

    "algorithm" : "auto"

}

model_knn_1 =  KNeighborsClassifier()

target_model_set["knn_1"] = [param_list_knn_1, model_knn_1]



param_list_rc_1 = {

    "alpha" : ["discrete_uniform",[0.1,1.0,0.01]],

    "random_state" : 1234

}

model_rc_1 =  RidgeClassifier()

target_model_set["rc_1"] = [param_list_rc_1, model_rc_1]
#params_for_stacking, parameter_search_results = mf.parameter_search(target_model_set=target_model_set, kf_num=5, trial_num=10, x_train=x_train, y_train=y_train, x_test=x_test)

params_for_stacking, parameter_search_results = mf.parameter_search(target_model_set=target_model_set, kf_num=5, trial_num=2, x_train=x_train, y_train=y_train, x_test=x_test)
for model_name, results in parameter_search_results.items():

    validation_result = results[1]

    print(model_name + " => " + validation_result)
# params_for_stacking.pop("lc_1")

params_for_stacking.keys()
stacking_result = mf.stacking_function(x_train=x_train, y_train=y_train, x_test=x_test, params_for_stacking=params_for_stacking, kf_num=5)
result_for_submission = [int(round(result)) for result in stacking_result]

submission['target'] = result_for_submission

submission.head()
submission.to_csv("submission.csv", index=False)