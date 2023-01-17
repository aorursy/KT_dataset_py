import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

print(os.listdir("../input/dataset-for-detection-of-cybertrolls/"))
df = pd.read_json('../input/dataset-for-detection-of-cybertrolls/Dataset for Detection of Cyber-Trolls.json', lines= True)

df.head()
df.shape
df["label"] = df.annotation.apply(lambda x: x.get('label'))

df["label"] = df.label.apply(lambda x: x[0])



df.head()
df.extras.unique()
df["notes"] = df.annotation.apply(lambda x: x.get('notes'))

df.notes.unique()
import nltk

nltk.download(['punkt', 'wordnet'])



import re

import numpy as np

import pandas as pd

from nltk.tokenize import word_tokenize

from nltk.stem import WordNetLemmatizer

from sklearn.metrics import confusion_matrix

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
def load_data(path):

    df = pd.read_json(path, lines= True)

    

    df["label"] = df.annotation.apply(lambda x: x.get('label'))

    df["label"] = df.label.apply(lambda x: x[0])

    

    X = df.content.values

    y = df.label.values

    

    return X, y



def tokenize(text):



    tokens = word_tokenize(text)

    lemmatizer = WordNetLemmatizer()



    clean_tokens = []

    for tok in tokens:

        clean_tok = lemmatizer.lemmatize(tok).lower().strip()

        clean_tokens.append(clean_tok)



    return clean_tokens



def display_results(y_test, y_pred):

    labels = np.unique(y_pred)

    confusion_mat = confusion_matrix(y_test, y_pred, labels=labels)

    accuracy = (y_pred == y_test).mean()



    print("Labels:", labels)

    print("Confusion Matrix:\n", confusion_mat)

    print("Accuracy:", accuracy)

    

def main():

    url = '../input/dataset-for-detection-of-cybertrolls/Dataset for Detection of Cyber-Trolls.json'

    X, y = load_data(url)

    X_train, X_test, y_train, y_test = train_test_split(X, y)



    vect = CountVectorizer(tokenizer=tokenize)

    tfidf = TfidfTransformer()

    clf = RandomForestClassifier()



    # train classifier

    X_train_counts = vect.fit_transform(X_train)

    X_train_tfidf = tfidf.fit_transform(X_train_counts)

    clf.fit(X_train_tfidf, y_train)



    # predict on test data

    X_test_counts = vect.transform(X_test)

    X_test_tfidf = tfidf.transform(X_test_counts)

    y_pred = clf.predict(X_test_tfidf)

    

    # predict on test data

    X_test_counts = vect.transform(["whoa stop you stupid sjw"])

    X_test_tfidf = tfidf.transform(X_test_counts)

    print("Given text: 'whoa stop you stupid sjw' ")

    print("Prediction: {}\n".format(clf.predict(X_test_tfidf)))

    



    # display results

    display_results(y_test, y_pred)





main()