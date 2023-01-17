# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
model_shape = {'input_dim': 200, 'embedding_len':300, 'hidden_size': 64}
model_settings = {'dropout': .5, 'learn_rate': .001}

train = True
if train:
    by_sentence = False
if train:
    labeled_data = pd.read_csv("../input/labeledTrainData.tsv", sep="\t")
    unlabeled_data = pd.read_csv("../input/unlabeledTrainData.tsv", sep="\t", quoting=3)
    labeled_data.head()
from bs4 import BeautifulSoup

def remove_html_tags(review_series):
    return review_series.apply(lambda x: BeautifulSoup(x, "lxml").get_text())
if train:
    labeled_data["text"] = remove_html_tags(labeled_data["review"])
    unlabeled_data["text"] = remove_html_tags(unlabeled_data["review"])

import spacy
nlp = spacy.blank("en")
nlp.add_pipe(nlp.create_pipe('sentencizer'))
from tqdm import tqdm
tqdm.pandas()

def to_spacy_doc(text_series):
    return text_series.progress_apply(lambda x: nlp(x))
if train:
    labeled_data["doc"] = to_spacy_doc(labeled_data["text"])
    unlabeled_data["doc"] = to_spacy_doc(unlabeled_data["text"])
    labeled_data["doc"].head()
if train:
    from sklearn.model_selection import train_test_split

    train_docs, test_docs, train_labels, test_labels = train_test_split(
                labeled_data["doc"], labeled_data["sentiment"].tolist(), 
                test_size=0.2, random_state=0)
if train and by_sentence:
    def get_labelled_sentences(docs, doc_labels):
        labels = []
        sentences = []
        for doc, y in zip(docs, doc_labels):
            for sent in doc.sents:
                sentences.append(sent)
                labels.append(y)
        return sentences, np.asarray(labels, dtype='int32')
    
    train_docs, train_labels = get_labelled_sentences(train_docs, train_labels)
if train:
    import collections

    dictionary = collections.Counter([token.lower_ for doc in train_docs for token in doc if not token.is_stop and not token.is_punct]) 
    dictionary.update([token.lower_ for doc in unlabeled_data["doc"].tolist() for token in doc if not token.is_stop and not token.is_punct])
    for i, word in enumerate(dictionary):
        dictionary[word] = i+1 # reserve 0 for unknown
def get_features(docs):
    docs = list(docs)
    features = np.zeros((len(docs), model_shape['input_dim']), dtype='int32')
    
    for i, doc in enumerate(docs):
        j = 0
        for token in doc:
            if not token.is_stop and not token.is_punct:
                if token.lower_ in dictionary:
                    features[i, j] = dictionary[token.lower_]

                j += 1
                if j >= model_shape['input_dim']:
                    break
    return features
if train:
    train_x = get_features(train_docs)
    test_x = get_features(test_docs)
from keras.optimizers import Adam

if train:
    from keras.models import Sequential
    from keras.layers import LSTM, Dense, Embedding, Bidirectional
    from keras.layers import TimeDistributed

    model = Sequential()
    model.add(Embedding(input_dim=len(dictionary)+1, 
                        output_dim=model_shape["embedding_len"], 
                        input_length=model_shape['input_dim'],
                        mask_zero=True))
    model.add(TimeDistributed(Dense(model_shape['hidden_size'], use_bias=False)))
    model.add(Bidirectional(LSTM(model_shape['hidden_size'],
                                 recurrent_dropout=model_settings['dropout'],
                                 dropout=model_settings['dropout'])))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer=Adam(lr=model_settings['learn_rate']), 
                  loss='binary_crossentropy', metrics=['accuracy'])
if train:
    model.fit(train_x, train_labels, epochs=5, batch_size=100)
import pickle

if train:
    with open('dictionary.pkl', 'wb') as pkl_file:
        pickle.dump(dictionary, pkl_file)
        
    with open("model.json", "w") as json_file:
        json_file.write(model.to_json())
    model.save_weights("model.h5")
if not train:
    from keras.models import model_from_json

    with open('dictionary.pkl', 'rb') as pkl_file:
        dictionary = pickle.load(pkl_file)
    
    with open('model.json', 'r') as json_file:
        model = model_from_json(json_file.read())
    model.load_weights("model.h5")
    model.compile(optimizer=Adam(lr=model_settings['learn_rate']), 
                  loss='binary_crossentropy', metrics=['accuracy'])
from sklearn.metrics import classification_report

def predict(model, x):
    y_pred = model.predict(x)
    return [1 if pred >= .5 else 0 for pred in y_pred]
def print_report(model, x, y_true):
    y_pred = predict(model, x)
    report = classification_report(y_true, y_pred)
    print(report)
if train:
    print_report(model, test_x, test_labels)
submission_data = pd.read_csv("../input/testData.tsv", sep="\t")
submission_data.head()
submission_data["text"] = remove_html_tags(submission_data["review"])
submission_data["doc"] = to_spacy_doc(submission_data["text"])
submission_data["doc"].head()
submission_x = get_features(submission_data["doc"].tolist())
submission_pred = predict(model, submission_x)
submission = pd.DataFrame({"id": submission_data["id"], "sentiment": submission_pred})
submission.head()
submission.to_csv("submission.csv", index=False)
