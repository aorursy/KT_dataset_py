import numpy as np

import pandas as pd

import seaborn as sb

import matplotlib.pyplot as plt



import sklearn as sk

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.svm import LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import cross_val_score, train_test_split

from sklearn.pipeline import Pipeline

from sklearn.metrics import accuracy_score, roc_auc_score

from sklearn.model_selection import cross_validate
o_train = pd.read_csv("../input/train.csv")

o_valid = pd.read_csv("../input/valid.csv")



train = pd.read_csv("../input/train.csv")

valid = pd.read_csv("../input/valid.csv")



data = pd.concat([train, valid], sort=False)



example_sub = pd.read_csv("../input/sample_submission.csv")
train.head(10)
train[train.ID == 5022].article_link
plt.figure(figsize=(10, 6))



sb.countplot(y='is_sarcastic', data=train)
sarcastic = len(train[train.is_sarcastic == 1])

non_sarcastic = len(train[train.is_sarcastic == 0])



sarcastic / (non_sarcastic + sarcastic)
print(np.where(pd.isnull(train)))

print(np.where(pd.isna(train)))

np.where(train.applymap(lambda x: x == ''))
sarcasm_classfication = Pipeline([

    ('vect', CountVectorizer()),

    ('tfidf', TfidfTransformer()),

    ('classify', LinearSVC(C=1))

])
X_train, X_test, y_train, y_test = train_test_split(train.headline, train.is_sarcastic)
sarcasm_classfication.fit(X_train, y_train)
print(roc_auc_score(y_train, sarcasm_classfication.decision_function(X_train)))

print(roc_auc_score(y_test, sarcasm_classfication.decision_function(X_test)))



cross_validate(sarcasm_classfication, train.headline, train.is_sarcastic, cv=5, scoring='roc_auc')
def remove_punctuation(dataframe):

    rgx = '(\'s|[!?,.:;\'$])'

    tmp = dataframe.copy()

    tmp['headline'] = tmp['headline'].str.replace(rgx, '')

    

    return tmp
tmp = train.copy()

tmp = remove_punctuation(tmp)



X_train, X_test, y_train, y_test = train_test_split(tmp.headline, tmp.is_sarcastic)



sarcasm_classfication.fit(X_train, y_train)



print(roc_auc_score(y_train, sarcasm_classfication.decision_function(X_train)))

print(roc_auc_score(y_test, sarcasm_classfication.decision_function(X_test)))



cross_validate(sarcasm_classfication, X_train, y_train, cv=5, scoring='roc_auc')
def get_top_n_words(corpus, n=None):

    vec = CountVectorizer().fit(corpus)

    bag_of_words = vec.transform(corpus)

    sum_words = bag_of_words.sum(axis=0) 

    words_freq = [(word, sum_words[0, idx], idx) for word, idx in vec.vocabulary_.items()]

    words_freq = sorted(words_freq, key = lambda x: x[1], reverse=True)

    

    words = [x[0] for x in words_freq]

    count = [x[1] for x in words_freq]

    

    return pd.DataFrame({'Words': words[:n], 'Amount': count[:n]})
plt.figure(figsize=(10, 6))



tmp = get_top_n_words(train.headline, 9)



sb.barplot(x=tmp.Words, y=tmp.Amount)
t = train.copy()

for word in tmp.Words:

    t.headline.str.replace(word, '')

    

X_train, X_test, y_train, y_test = train_test_split(t.headline, t.is_sarcastic)



sarcasm_classfication.fit(X_train, y_train)



print(roc_auc_score(y_train, sarcasm_classfication.decision_function(X_train)))

print(roc_auc_score(y_test, sarcasm_classfication.decision_function(X_test)))



cross_validate(sarcasm_classfication, X_train, y_train, cv=5, scoring='roc_auc')
def remove_double_links(series):

    rgx = '(https?(?!.+https?).+)'

    

    tmp = series.copy()

    tmp['article_link'] = tmp['article_link'].str.extract(rgx)

    

    return tmp
def get_source(series):

    rgx = '((?!https?:)\/\/.+?\..+?\/)'

    

    tmp = series.copy()

    tmp['article_link'] = tmp['article_link'].str.extract(rgx)

    tmp['article_link'] = tmp['article_link'].str.strip(to_strip="/w")

    tmp['article_link'] = tmp['article_link'].str.strip(to_strip=r'^\.')

    return tmp
train = remove_double_links(train)

train = get_source(train)

train.groupby('article_link')['article_link'].describe()
def rename_article_link(dataframe):

    return dataframe.rename(columns={'article_link': 'source'})
train = rename_article_link(train)

train.head(10)
train['headline_and_source'] = train.source + ' ' + train.headline
train.head(10)
X_train, X_test, y_train, y_test = train_test_split(train['headline_and_source'], train.is_sarcastic)



sarcasm_classfication.fit(X_train, y_train)
print(roc_auc_score(y_train, sarcasm_classfication.decision_function(X_train)))

print(roc_auc_score(y_test, sarcasm_classfication.decision_function(X_test)))

cross_validate(sarcasm_classfication, train.headline_and_source, train.is_sarcastic, cv=25, scoring='roc_auc', return_train_score=True)
tmp = train.copy()



X_tmp_train, X_tmp_test, y_tmp_train, y_tmp_test = train_test_split(tmp.source, tmp.is_sarcastic)



tmp_model = sarcasm_classfication



tmp_model.fit(X_tmp_train, y_tmp_train)



print(roc_auc_score(y_tmp_train, tmp_model.decision_function(X_tmp_train)))

print(roc_auc_score(y_tmp_test, tmp_model.decision_function(X_tmp_test)))



cross_validate(sarcasm_classfication, X_tmp_train, y_tmp_train, cv=5, scoring='roc_auc', return_train_score=True)
tmp = train.copy()



tmp = tmp.drop(columns=['ID', 'headline', 'headline_and_source'])



tmp = pd.get_dummies(data=tmp, columns=['source'])



X_tmp_train, X_tmp_test, y_tmp_train, y_tmp_test = train_test_split(tmp.drop(columns=['is_sarcastic']), tmp.is_sarcastic)



tmp_model = RandomForestClassifier(n_jobs=-1, n_estimators=100)



tmp_model.fit(X_tmp_train, y_tmp_train)



prob = tmp_model.predict_proba(X_tmp_train)

prob = prob[:, 1]



print(roc_auc_score(y_tmp_train, prob))



prob = tmp_model.predict_proba(X_tmp_test)

prob = prob[:, 1]



print(roc_auc_score(y_tmp_test, prob))



cross_validate(tmp_model, X_tmp_train, y_tmp_train, cv=5, scoring='roc_auc', return_train_score=True)
valid = remove_double_links(valid)

valid = get_source(valid)

valid = valid.rename(columns={'article_link': 'source'})
predicted = sarcasm_classfication.predict(valid.source)



prediction_dataframe = pd.DataFrame({'ID': valid.ID, 'is_sarcastic': predicted})



prediction_dataframe.to_csv('output.csv', index=False)