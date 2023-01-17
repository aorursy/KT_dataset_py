import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
input_path = '/kaggle/input/nlp-getting-started/'
train = pd.read_csv(os.path.join(input_path, 'train.csv'))
test = pd.read_csv(os.path.join(input_path, 'test.csv'))
train.head()
print('Train: ', train.shape)
print('Test: ', test.shape)
# check the missing values for keyword and location
len(train['keyword'].isnull()), len(train['location'].isnull())
# non disaster tweet
train[train['target'] == 0]['text'].values[0]
# disaster tweet
train[train['target'] == 1]['text'].values[0]
import re
import unicodedata
import spacy
from sklearn.model_selection import cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
# reference:~ https://github.com/dipanjanS/practical-machine-learning-with-python/blob/master/bonus%20content/nlp%20proven%20approach/contractions.py

CONTRACTION_MAP = {
    "ain't": "is not",
    "aren't": "are not",
    "can't": "cannot",
    "can't've": "cannot have",
    "'cause": "because",
    "could've": "could have",
    "couldn't": "could not",
    "couldn't've": "could not have",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hadn't've": "had not have",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he would",
    "he'd've": "he would have",
    "he'll": "he will",
    "he'll've": "he he will have",
    "he's": "he is",
    "how'd": "how did",
    "how'd'y": "how do you",
    "how'll": "how will",
    "how's": "how is",
    "I'd": "I would",
    "I'd've": "I would have",
    "I'll": "I will",
    "I'll've": "I will have",
    "I'm": "I am",
    "I've": "I have",
    "i'd": "i would",
    "i'd've": "i would have",
    "i'll": "i will",
    "i'll've": "i will have",
    "i'm": "i am",
    "i've": "i have",
    "isn't": "is not",
    "it'd": "it would",
    "it'd've": "it would have",
    "it'll": "it will",
    "it'll've": "it will have",
    "it's": "it is",
    "let's": "let us",
    "ma'am": "madam",
    "mayn't": "may not",
    "might've": "might have",
    "mightn't": "might not",
    "mightn't've": "might not have",
    "must've": "must have",
    "mustn't": "must not",
    "mustn't've": "must not have",
    "needn't": "need not",
    "needn't've": "need not have",
    "o'clock": "of the clock",
    "oughtn't": "ought not",
    "oughtn't've": "ought not have",
    "shan't": "shall not",
    "sha'n't": "shall not",
    "shan't've": "shall not have",
    "she'd": "she would",
    "she'd've": "she would have",
    "she'll": "she will",
    "she'll've": "she will have",
    "she's": "she is",
    "should've": "should have",
    "shouldn't": "should not",
    "shouldn't've": "should not have",
    "so've": "so have",
    "so's": "so as",
    "that'd": "that would",
    "that'd've": "that would have",
    "that's": "that is",
    "there'd": "there would",
    "there'd've": "there would have",
    "there's": "there is",
    "they'd": "they would",
    "they'd've": "they would have",
    "they'll": "they will",
    "they'll've": "they will have",
    "they're": "they are",
    "they've": "they have",
    "to've": "to have",
    "wasn't": "was not",
    "we'd": "we would",
    "we'd've": "we would have",
    "we'll": "we will",
    "we'll've": "we will have",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what'll": "what will",
    "what'll've": "what will have",
    "what're": "what are",
    "what's": "what is",
    "what've": "what have",
    "when's": "when is",
    "when've": "when have",
    "where'd": "where did",
    "where's": "where is",
    "where've": "where have",
    "who'll": "who will",
    "who'll've": "who will have",
    "who's": "who is",
    "who've": "who have",
    "why's": "why is",
    "why've": "why have",
    "will've": "will have",
    "won't": "will not",
    "won't've": "will not have",
    "would've": "would have",
    "wouldn't": "would not",
    "wouldn't've": "would not have",
    "y'all": "you all",
    "y'all'd": "you all would",
    "y'all'd've": "you all would have",
    "y'all're": "you all are",
    "y'all've": "you all have",
    "you'd": "you would",
    "you'd've": "you would have",
    "you'll": "you will",
    "you'll've": "you will have",
    "you're": "you are",
    "you've": "you have"
}
# loading the spacy's en_core_web_sm
nlp = spacy.load('en_core_web_sm')
nlp.pipe_names
# create and add sentencizer to the pipeline
sent = nlp.create_pipe('sentencizer')
nlp.add_pipe(sent, before='parser')
nlp.pipe_names
def text_cleaning(text):
    """
    Returns cleaned text (Accented Characters, Expand Contractions, Special Characters)
    Parameters
    ----------
    text -> String
    """
    # remove accented characters
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    
    # expand contractions
    for word in text.split():
        if word.lower() in CONTRACTION_MAP:
            text = text.replace(word[1:], CONTRACTION_MAP[word.lower()][1:])
    
    # remove special characters
    pattern = r'[^a-zA-Z0-9\s,:)(!]'
    text = re.sub(pattern, '', text)
    
    doc = nlp(text)
    tokens = []
    
    for token in doc:
        if token.lemma_ != '-PRON-':
            tokens.append(token.lemma_.lower().strip())
        else:
            tokens.append(token.lower_)

    return tokens
text_cleaning("I don't like this movie :)")
# split the data into inputs and outputs
X_train = train['text']
y_train = train['target']
X_test = test['text']
# build a model pipeline
# stage 1: preprocessing, stage 2: linear SVC

text_clf = Pipeline([
    ('tfidf', TfidfVectorizer(tokenizer=text_cleaning)),
    ('clf', LinearSVC())
])
# using F1 metric in cross validation
scores = cross_val_score(text_clf, X_train, y_train, cv=3, scoring='f1')
scores
# fit the model on train data
text_clf.fit(X_train, y_train)
# load the sample submission csv file
sample_submission = pd.read_csv(os.path.join(input_path, 'sample_submission.csv'))
# predict on test data
sample_submission['target'] = text_clf.predict(X_test)
# save the sample submission csv file
sample_submission.to_csv('submission.csv', index=False)
