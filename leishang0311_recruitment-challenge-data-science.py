import os
print(os.listdir())
#print(os.listdir("../input"))

import glob
import pandas as pd
import csv
import string
import nltk
#from autocorrect import spell
import time

import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, precision_recall_fscore_support, classification_report
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier

import sys
print(sys.version)
# Dataset config
SUBSET_MAP = {
    'Yelp': 'train',
    'IMDB': 'train',
    'Amazon': 'test'
}

SUBSET_SIZE = {
    'train': 2000,  
    'test': 1000  
}

LABEL_MAP = {
    1: 'positive',
    0: 'negative'
}

path = r'../input/sentimentanalysis'
print(os.listdir(path))
input_files = glob.glob(path + "/*_labelled.txt")
print("Read input files {}...".format(input_files))
list_ = []
for file_ in input_files:
    df_ = pd.read_csv(
        file_,
        delimiter='\t',
        quoting=csv.QUOTE_NONE,
        header=None,
        names=['text', 'label']
    )
    df_['source'] = file_
    list_.append(df_)
df = pd.concat(list_, ignore_index=True)
assert df.shape == (sum(SUBSET_SIZE.values()), 2 + 1)

def split_data(row):
    subset = 'UNKNOWN'
    for item in SUBSET_MAP.items():
        if item[0].lower() in row[-1].lower():
            subset = item[1]
    return subset

df['subset'] = df.apply(split_data, axis=1)
display(df.head(5))
display(df.tail(5))
assert set(SUBSET_MAP.values()) == set(df['subset'].unique())

print("Pre-processing...")
start_time = time.time()

df['text'] = df['text'].apply(lambda x: x.lower())  # lower case

table = str.maketrans({}.fromkeys(string.punctuation))  # remove punctuation
df['text'] = df['text'].apply(lambda x: x.translate(table))

df['text'] = df['text'].apply(nltk.word_tokenize)  # tokenize

def spell_check(text):
    checked = []
    for word in text:
        checked.append(spell(word))
    return checked

# fix typos (disabled since no significant improvement in model performance)
#df['text'] = df['text'].apply(spell_check)  # fix typos
#print(df.head())

# stopwords removal (disabled since no improvement in model performance)
# from nltk.corpus import stopwords
# stop = stopwords.words('english')
# df['text'] = df['text'].apply(lambda x: [item for item in x if item not in stop])

# stemming (disabled since no improvement in model performance)
# ps = nltk.stem.PorterStemmer()
# df['text'] = df['text'].apply(lambda x: [ps.stem(y) for y in x])

print("time taken: {0:.2f} seconds".format(time.time() - start_time))
mask = df['subset'].str.contains('train')
df_train = df[mask]
df_test = df[~mask]

assert df_train.shape[0] == SUBSET_SIZE['train']
assert df_test.shape[0] == SUBSET_SIZE['test']

assert len(df_train['label'].unique()) == 2
assert len(df_test['label'].unique()) == 2

# class distribution
print('training set class distribution:')
print(df_train['label'].value_counts(normalize=True))

print('test set class distribution:')
print(df_test['label'].value_counts(normalize=True))

# The number of instances of the two classes are equal
# We conclude that there is no Class Imbalanced Problem
style = 'bmh'
with plt.style.context(style):
    fig = plt.figure(figsize=(8, 5))

    color = 'blue'
    layout = (1, 2)

    ax1 = plt.subplot2grid(layout, (0, 0))
    df_train['label'].value_counts().plot(kind='bar', ax=ax1)
    ax1.set_title('Count of labels - Train')
    plt.xticks(rotation=0)

    ax2 = plt.subplot2grid(layout, (0, 1), sharey=ax1)
    df_test['label'].value_counts().plot(kind='bar', ax=ax2)
    ax2.set_title('Count of labels - Test')
    plt.xticks(rotation=0)

#Q2a soln.
flat_word_list = [item for sublist in df_train['text'] for item in sublist]
print(flat_word_list[1:10])
word_counts = pd.Series(flat_word_list).value_counts()
top_n = 10  # top n words
print("{} most frequent words".format(top_n))
display(
    pd.DataFrame(
        {
            'word': word_counts[:top_n].index,
            'frequency': word_counts[:top_n].values
        }
    )
)
import codecs
from tqdm import tqdm
import numpy as np

# load the model
f = codecs.open("../input/wikinews300d1mvec/wiki-news-300d-1M.vec", encoding='utf8')
fasttext_model = {}
for line in tqdm(f):
    values = line.rstrip().rsplit(' ')
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    fasttext_model[word] = coefs
f.close()
print('found %s word vectors' % len(fasttext_model))
# function to convert a sentence to vector
def sentence2vector(sentence, model):
    vector = []
    num_words = 0
    for word in sentence:
        try:
            if num_words == 0:
                vector = model[word]
            else:
                vector = np.add(vector, model[word])
            num_words += 1
        except:
            #print("{} not found".format(word))
            pass

    assert len(vector) != 0
    return np.asarray(vector) / num_words

# convert text samples to vectors
print('convert text to vectors...')
start_time = time.time()

X_train = []
for i, v in enumerate(df_train['text'].tolist()):
    X_train.append(sentence2vector(v, fasttext_model))

X_test = []
for i, v in enumerate(df_test['text'].tolist()):
    X_test.append(sentence2vector(v, fasttext_model))
    
print("time taken: {0:.2f} seconds".format(time.time() - start_time))
  
y_train = df_train['label']
y_test = df_test['label']

assert len(X_train) == 2000
assert len(X_test) == 1000
start_time = time.time()
clf = GridSearchCV(
    MLPClassifier(max_iter=400, random_state=123),
    param_grid={
        'alpha': [0.1, 0.5],
        'solver': ['adam']
    },
    cv=2
)
clf.fit(X_train, y_train)
print("time taken: {0:.2f} seconds".format(time.time() - start_time))

print(clf.best_estimator_)
#Q4 soln.
print("Evaluate ...")
print('Accuracy:', clf.score(X_test, y_test))

print('Confusion matrix:')
import collections
y_pred = clf.predict(X_test)
print("Prediction: {}".format(collections.Counter(y_pred)))
print("Actual: {}".format(collections.Counter(y_test)))
print(confusion_matrix(y_test, clf.predict(X_test)))

y_pred = clf.predict(X_test)

print('Classification report:')
print(classification_report(y_test, clf.predict(X_test)))

fig = plt.figure()
precision, recall, _ = precision_recall_curve(y_test, [p[1] for p in clf.predict_proba(X_test)])
plt.plot(recall, precision)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision and Recall Curve')
plt.show()
fig = plt.figure()
y_pred_proba = clf.predict_proba(X_test)[::, 1]
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
auc = roc_auc_score(y_test, y_pred_proba)

plt.title('ROC Curve')
plt.plot(fpr, tpr, linewidth=3)
plt.plot([0, 1], [0, 1], 'k--')
plt.axis([-0.005, 1, 0, 1.005])
plt.plot(fpr, tpr, label="ROC Curve, auc= %0.3f" % auc)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate (Recall)")
plt.legend(loc='best')
plt.show()


from sklearn.metrics import make_scorer, recall_score, precision_score
start_time = time.time()
clf = GridSearchCV(
    MLPClassifier(max_iter=400, random_state=123),
    scoring=make_scorer(precision_score),
    param_grid={
        'alpha': [0.1, 0.5],
        'solver': ['adam']
    },
    cv=2
)
clf.fit(X_train, y_train)
print("time taken: {0:.2f} seconds".format(time.time() - start_time))

print(clf.best_estimator_)
print("Prediction: {}".format(collections.Counter(y_pred)))
print("Actual: {}".format(collections.Counter(y_test)))

print('Confusion matrix:')
print(confusion_matrix(y_test, clf.predict(X_test)))

y_scores = clf.predict_proba(X_test)[:,1]

threshold = 0.3   # decision threshold
y_pred = [1 if y >= threshold else 0 for y in y_scores]

print("Prediction: {}".format(collections.Counter(y_pred)))
print("Actual: {}".format(collections.Counter(y_test)))

print('Accuracy:', accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

print('Classification report:')
print(classification_report(y_test, y_pred))