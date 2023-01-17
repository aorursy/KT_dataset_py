# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
%pylab inline

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
quora_train = pd.read_csv("../input/train.csv")
quora_train.head()
quora_train.info()
print(f"Number of neitral texts: {len(quora_train[quora_train['target']==0])}")
print(f"Number of neitral texts: {len(quora_train[quora_train['target']==1])}")
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
vect = TfidfVectorizer()
sklearn_tokenizer = vect.build_tokenizer()
sklearn_tokenizer(quora_train.loc[2]["question_text"])
word_tokenize(quora_train.loc[2]["question_text"])
from nltk.corpus import stopwords

stopwords.words("english")
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import StratifiedKFold

vect = TfidfVectorizer(stop_words=stopwords.words("english"))
clf = SGDClassifier(loss="modified_huber", max_iter=6, class_weight = "balanced")
model = Pipeline([('vect', vect), ('clf', clf)])
%%time
preds = cross_val_predict(model, quora_train.question_text.values, quora_train.target.values,
                          cv=StratifiedKFold(4), n_jobs=-1,
                          method='predict_proba')
from sklearn.metrics import roc_auc_score, classification_report, f1_score

roc_auc_score(quora_train.target.values, preds[:,1])
thresholds = np.arange(0.05, 0.95, 0.05)
pred_arr = []
for threshold in thresholds:
    pred_arr.append(list(map(lambda x: 1 if x[1]>threshold else 0, preds)))
i = 0
for pred in pred_arr:
    print(f"F1-score = {f1_score(quora_train.target.values, np.array(pred))} with threshold = {thresholds[i]}")
    i += 1
best_predictions = list(map(lambda x: 1 if x[1]>0.7 else 0, preds))
print(classification_report(quora_train.target.values, best_predictions))
import eli5
model.fit(quora_train.question_text.values, quora_train.target.values)
eli5.show_weights(model, top=20)
from nltk.stem import SnowballStemmer, WordNetLemmatizer, LancasterStemmer
from functools import lru_cache
@lru_cache(maxsize=2048)
def lemmatize_word(word):
    parts = ['a','v','n','r']
    lemmatizer = WordNetLemmatizer()
    for part in parts:
        temp = lemmatizer.lemmatize(word, part)
        if temp != word:
            return temp
    return word    
stemmer = SnowballStemmer('english')
print(lemmatize_word('evening'))
print(stemmer.stem('evening'))
def lemmatize_sentence(sentence, tokinizer):
    return list(map(lemmatize_word, tokinizer(sentence)))

def stemmatize_sentence(sentence, tokinizer):
    stemmer = SnowballStemmer('english')
    return list(map(stemmer.stem, tokinizer(sentence)))
%%time

lemmatized_data = list(map(lambda t: lemmatize_sentence(t, sklearn_tokenizer), quora_train.question_text.values))
stemmatized_data = list(map(lambda t: stemmatize_sentence(t, sklearn_tokenizer), quora_train.question_text.values))
%%time

inv_lemmatized_data = list(map(lambda t: " ".join(t), lemmatized_data))
inv_stemmatized_data = list(map(lambda t: " ".join(t), stemmatized_data))
stemming_preds = cross_val_predict(model, inv_stemmatized_data, quora_train.target.values,
                          cv=StratifiedKFold(4), n_jobs=4, verbose=1,
                          method='predict_proba')
lemming_preds = cross_val_predict(model, inv_lemmatized_data, quora_train.target.values,
                          cv=StratifiedKFold(4), n_jobs=4, verbose=1,
                          method='predict_proba')
lemming_pred_arr = []
stemming_pred_arr = []
for threshold in thresholds:
    lemming_pred_arr.append(list(map(lambda x: 1 if x[1]>threshold else 0, lemming_preds)))
    stemming_pred_arr.append(list(map(lambda x: 1 if x[1]>threshold else 0, stemming_preds)))
i = 0
print("Lemmatization")
for pred in lemming_pred_arr:
    print(f"F1-score = {f1_score(quora_train.target.values, pred)} with threshold = {thresholds[i]}")
    i += 1
i = 0
print("Stemming")
for pred in stemming_pred_arr:
    print(f"F1-score = {f1_score(quora_train.target.values, pred)} with threshold = {thresholds[i]}")
    i += 1
model.get_params().keys()
# зададим сетку параметров для перебора модельки
"""Я уже перебрал(207 минут!!!), получил такие лучшие параметры, 
{'clf__l1_ratio': 0.15,
 'clf__loss': 'modified_huber',
 'clf__max_iter': 3,
 'vect__max_df': 0.8,
 'vect__min_df': 0,
 'vect__stop_words': None}"""
"""
Вот старая сетка параметров. Теперь переберу быстрее, чтобы дебильный kernel смог закоммитить
param_grid = {
    'clf__l1_ratio': [0.15, 0.35],
    'clf__loss': ['log', 'modified_huber'],
    'clf__max_iter': [3, 5, 7],
    'vect__stop_words': [None, stopwords.words("english")],
    'vect__max_df': [0.8, 1.0],
    'vect__min_df': [0, 100]
}"""

param_grid = {
    'clf__max_iter': [2, 3],
    'vect__max_df': [0.7, 0.8]
}
from sklearn.model_selection import GridSearchCV

new_vect = TfidfVectorizer(lowercase=True)
new_clf = SGDClassifier(class_weight = "balanced", loss='modified_huber')
new_model = Pipeline([('vect', new_vect), ('clf', new_clf)])
realy_long_search = GridSearchCV(new_model, param_grid, cv = StratifiedKFold(4), verbose=1, n_jobs=4, scoring='f1')
realy_long_search.fit(inv_stemmatized_data, quora_train.target.values)
realy_long_search.best_params_
best_sgd = realy_long_search.best_estimator_
best_predictions = cross_val_predict(best_sgd, inv_stemmatized_data, quora_train.target.values,
                          cv=StratifiedKFold(4), n_jobs=4, verbose=1,
                          method='predict_proba')
best_pred_arr = []
for threshold in thresholds:
    best_pred_arr.append(list(map(lambda x: 1 if x[1]>threshold else 0, best_predictions)))
i = 0
print("Best model scores")
for pred in best_pred_arr:
    print(f"F1-score = {f1_score(quora_train.target.values, pred)} with threshold = {thresholds[i]:.2f}")
    i += 1
best_threshold = 0.75
submission_example = pd.read_csv("../input/sample_submission.csv")
submission_example.head()
test = pd.read_csv("../input/test.csv")
test.head()
%%time

test_stemmatized_data = list(map(lambda t: stemmatize_sentence(t, sklearn_tokenizer), test.question_text.values))
test_inv_stemmatized_data = list(map(lambda t: " ".join(t), test_stemmatized_data))
test_predictions = best_sgd.predict_proba(test_inv_stemmatized_data)
out_predictions = list(map(lambda x: 1 if x[1]>best_threshold else 0, test_predictions))
out_predictions[200:220]
answers = pd.DataFrame(np.transpose([test["qid"].values, out_predictions]), columns=["qid", "prediction"])
answers.head()
answers.to_csv('submission.csv',index=False)
len(answers)
"""from IPython.display import HTML
import base64

# function that takes in a dataframe and creates a text link to  
# download it (will only work for files < 2MB or so)
def create_download_link(df, title = "Download CSV file", filename = "data.csv"):  
    csv = df.to_csv()
    b64 = base64.b64encode(csv.encode())
    payload = b64.decode()
    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'
    html = html.format(payload=payload,title=title,filename=filename)
    return HTML(html)

# create a random sample dataframe

# create a link to download the dataframe
create_download_link(answers.iloc[30000:])"""

