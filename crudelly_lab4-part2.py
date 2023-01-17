# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
quora_train = pd.read_csv("../input/train.csv")
quora_train.head()
from nltk.stem import SnowballStemmer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score


def stemmatize_sentence(sentence, tokinizer):
    stemmer = SnowballStemmer('english')
    return list(map(stemmer.stem, tokinizer(sentence)))
vect = TfidfVectorizer(lowercase=True, max_df=0.8)
tokenizer = vect.build_tokenizer()
clf = SGDClassifier(class_weight = "balanced", loss='modified_huber', max_iter=2)
sgd_model = Pipeline([('vect', vect), ('clf', clf)])


%%time

stemmatized_data = list(map(lambda t: stemmatize_sentence(t, tokenizer), quora_train.question_text.values))
train_X = list(map(lambda t: " ".join(t), stemmatized_data))
train_y = quora_train.target.values
np.array(train_X).shape, np.array(train_y).shape
train_X[4], train_y[4]
%%time

sgd_pred_prob = cross_val_predict(sgd_model, train_X, train_y,
                          cv=StratifiedKFold(4, random_state=42), n_jobs=1, verbose=1,
                          method='predict_proba')
thresholds = np.arange(0.05, 1, 0.05)
sgd_pred_arr = []
for threshold in thresholds:
    sgd_pred_arr.append(list(map(lambda x: 1 if x[1]>threshold else 0, sgd_pred_prob)))
i = 0
print("SGD with stemming")
for pred in sgd_pred_arr:
    print(f"F1-score = {f1_score(train_y, pred)} with threshold = {thresholds[i]}")
    i += 1
from gensim.models import KeyedVectors

w2v = KeyedVectors.load_word2vec_format('../input/embeddings/GoogleNews-vectors-negative300/GoogleNews-vectors-negative300.bin', binary=True)
w2v["Obama"]
w2v.most_similar(positive=['Minsk', 'Russia'], negative=['Belarus'])
w2v.most_similar(positive=['Minsk', 'Ukrain'], negative=['Belarus'])
def w2v_sentence(sentence, tokenizer):
    word_list = tokenizer(sentence)
    w2v_vector = []
    for word in word_list:
        if word in w2v.vocab:
            w2v_vector.append(w2v[word])
        else:
            pass
    if len(w2v_vector) == 0:
        return np.zeros(300)
    else:
        return np.mean(w2v_vector, axis=0)
    
%%time

w2v_data = list(map(lambda t: w2v_sentence(t, tokenizer), quora_train.question_text.values))
del w2v # чтобы освободить ram, а то кагля умирает :(
w2v_data[4]
# Воспользуемся той же моделькой
w2v_pred_prob = cross_val_predict(clf, w2v_data, quora_train.target.values,
                          cv=StratifiedKFold(4, random_state=42), n_jobs=1, verbose=1,
                          method='predict_proba')
w2v_pred_arr = []
for threshold in thresholds:
    w2v_pred_arr.append(list(map(lambda x: 1 if x[1]>threshold else 0, w2v_pred_prob)))

i = 0
print("SGD with w2v")
for pred in w2v_pred_arr:
    print(f"F1-score = {f1_score(quora_train.target.values, pred)} with threshold = {thresholds[i]}")
    i += 1

new_X = list(zip(w2v_pred_prob[:,1], sgd_pred_prob[:,1]))
new_X[500]

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression

"""log_reg_grid = {
    'max_iter': [60, 80, 100, 150]
}

final_search = GridSearchCV(LogisticRegression(class_weight='balanced', solver="lbfgs"), log_reg_grid,
                           cv=StratifiedKFold(4, random_state=42), n_jobs=4, verbose=1,
                           scoring='f1')"""
# final_search.fit(new_X, train_y)
best_log_reg = LogisticRegression(max_iter=100, class_weight='balanced', solver="lbfgs")
final_pred_prob = cross_val_predict(best_log_reg, new_X, quora_train.target.values,
                          cv=StratifiedKFold(4, random_state=42), n_jobs=4, verbose=1,
                          method='predict_proba')
final_pred_arr = []
for threshold in thresholds:
    final_pred_arr.append(list(map(lambda x: 1 if x[1]>threshold else 0, final_pred_prob)))

i = 0
print("Final model")
for pred in final_pred_arr:
    print(f"F1-score = {f1_score(quora_train.target.values, pred)} with threshold = {thresholds[i]}")
    i += 1
quora_test = pd.read_csv("../input/test.csv")
sgd_model.fit(train_X, train_y)
w2v_model = SGDClassifier(class_weight = "balanced", loss='modified_huber', max_iter=2)
w2v_model.fit(w2v_data, train_y)
del w2v_data
del train_X
del train_y
del quora_train
w2v = KeyedVectors.load_word2vec_format('../input/embeddings/GoogleNews-vectors-negative300/GoogleNews-vectors-negative300.bin', binary=True)
w2v_test = list(map(lambda t: w2v_sentence(t, tokenizer), quora_test.question_text.values))
del w2v
quora_train = pd.read_csv("../input/train.csv").target.values
best_log_reg.fit(new_X, quora_train)
del quora_train
test1 = sgd_model.predict_proba(quora_test.question_text.values)[:,1]
test2 = w2v_model.predict_proba(w2v_test)[:,1]
answers_prob = best_log_reg.predict_proba(list(zip(test1, test2)))
answers = list(map(lambda x: 1 if x[1]>0.85 else 0, answers_prob))
out_df = pd.DataFrame(np.transpose([quora_test.qid, answers]), columns=["qid", "prediction"])
out_df.head()
out_df.to_csv("submission.csv", index=False)
