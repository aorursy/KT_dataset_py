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
labeled_data = pd.read_csv("../input/labeledTrainData.tsv", sep="\t")
unlabeled_data = pd.read_csv("../input/unlabeledTrainData.tsv", sep="\t", quoting=3)
labeled_data.head()
print(labeled_data.shape)
print(unlabeled_data.shape)
labeled_data["review"][0]
import bs4

def remove_html_tags(review_series):
    return review_series.apply(lambda x: bs4.BeautifulSoup(x, "lxml").get_text())
labeled_data["text"] = remove_html_tags(labeled_data["review"])
unlabeled_data["text"] = remove_html_tags(unlabeled_data["review"])
labeled_data["text"][0]
import spacy
nlp = spacy.blank("en")
from tqdm import tqdm
tqdm.pandas()

def get_tokens(text_series):
    return text_series.progress_apply(lambda x: [word.lower_ for word in nlp(x)
                                                 if not word.is_stop and not word.is_punct])
labeled_data["tokens"] = get_tokens(labeled_data["text"])
unlabeled_data["tokens"] = get_tokens(unlabeled_data["text"])
labeled_data["tokens"].head()
from sklearn.model_selection import train_test_split

train_tokens, test_tokens, train_labels, test_labels = train_test_split(
            labeled_data["tokens"].tolist(), labeled_data["sentiment"].tolist(), 
            test_size=0.2, random_state=0)
from gensim.models.word2vec import Word2Vec

n_feat = 300
tokens = train_tokens + unlabeled_data["tokens"].tolist()

w2v = Word2Vec(tokens, size=n_feat)
w2v.most_similar("good")
w2v.doesnt_match(["good", "decent", "popcorn", "great"])
from sklearn.preprocessing import scale # center the mean

def get_avg_wv(tokens_series, n_feat, w2v):
    vec = np.zeros((len(tokens_series), 1, n_feat))
    for i, tokens in enumerate(tokens_series):
        count = 0.
        for word in tokens:
            try:
                vec[i, :] += w2v[word].reshape(1, n_feat)
                count += 1.
            except KeyError: # handling the case where the token is not in the corpus
                continue
                
        if count != 0:
            vec[i, :] /= count
    return scale(np.concatenate(vec))
train_avg_wv = get_avg_wv(train_tokens, n_feat, w2v)
test_avg_wv = get_avg_wv(test_tokens, n_feat, w2v)
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(analyzer=lambda x: x)
_ = vectorizer.fit_transform(train_tokens)
tfidf = dict(zip(vectorizer.get_feature_names(), vectorizer.idf_))
print('vocab size :', len(tfidf))

def get_weighted_wv(tokens_series, n_feat, w2v):
    vec = np.zeros((len(tokens_series), 1, n_feat))
    for i, tokens in enumerate(tokens_series):
        count = 0.
        for word in tokens:
            try:
                vec[i, :] += w2v[word].reshape((1, n_feat)) * tfidf[word]
                count += 1.
            except KeyError: # handling the case where the token is not in the corpus
                continue
        if count != 0:
            vec[i, :] /= count
    return scale(np.concatenate(vec))
train_weighted_wv = get_weighted_wv(train_tokens, n_feat, w2v)
test_weighted_wv = get_weighted_wv(test_tokens, n_feat, w2v)
from keras.models import Sequential
from keras.layers import Dense, Dropout

def get_model(n_feat):
    model = Sequential()
    model.add(Dense(32, activation='relu', input_dim=n_feat))
    model.add(Dense(128, activation="relu"))
    #model.add(Dropout(.2))
    model.add(Dense(256, activation="relu"))
    #model.add(Dropout(.5))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(optimizer="rmsprop", loss="binary_crossentropy", metrics=["accuracy"])
    return model
avg_model = get_model(n_feat)
avg_model.fit(train_avg_wv, train_labels, epochs=10)
weighted_model = get_model(n_feat)
weighted_model.fit(train_weighted_wv, train_labels, epochs=10)
from sklearn.metrics import classification_report

def predict(model, x):
    y_pred = model.predict(x)
    return [1 if pred >= .5 else 0 for pred in y_pred]
def print_report(model, x, y_true):
    y_pred = predict(model, x)
    report = classification_report(y_true, y_pred)
    print(report)
print_report(avg_model, test_avg_wv, test_labels)
print_report(weighted_model, test_weighted_wv, test_labels)
submission_data = pd.read_csv("../input/testData.tsv", sep="\t")
submission_data.head()
submission_data["text"] = remove_html_tags(submission_data["review"])
submission_data["tokens"] = get_tokens(submission_data["text"])
submission_data["tokens"].head()
submission_weighted_wv = get_weighted_wv(submission_data["tokens"].tolist(), n_feat, w2v)
submission_pred = predict(weighted_model, submission_weighted_wv)
submission = pd.DataFrame({"id": submission_data["id"], 
                           "sentiment": submission_pred})
submission.head()
submission.to_csv("submission.csv", index=False)
