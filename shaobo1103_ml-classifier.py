import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from time import time
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier as SC, LogisticRegression as LR

from sklearn.metrics import f1_score
from sklearn.utils import shuffle

import joblib
train_file_path = '../input/news-classification/Data/train_set.csv'
test_file_path = '../input/news-classification/Data/test_a.csv'
train_df = pd.read_csv(train_file_path, sep='\t')
train_df.shape
def balance(df: pd.DataFrame):
    df = df.append(df[df['label']==11], ignore_index=True)
    tmp_df = pd.concat([df[df['label']==12]]*2, ignore_index=True)
    df = df.append(tmp_df, ignore_index=True)
    tmp_df = pd.concat([df[df['label']==13]]*4, ignore_index=True)
    df = df.append(tmp_df, ignore_index=True)
    return df


train_df = balance(train_df)
train_df = shuffle(train_df)
train_df.shape
max_len = 600
n_gram = (1,3)
max_features = 5000
tfidf = TfidfVectorizer(ngram_range=n_gram, max_features=max_features)

def pad_sequences(line: str):
    if len(line) >= max_len:
        line = line[:max_len]
    else:
        line += ' 0' * (max_len-len(line))
    return line


def feature_engineer():
    t0 = time()
    train_df['text'] = train_df['text'].apply(pad_sequences)
    train_test = tfidf.fit_transform(train_df['text'])
    print('processing time(s):', time()-t0)
    return train_test
    
train_test = feature_engineer()
def training(clf, n_rows, name):
    t0 = time()
    clf.fit(train_test[:n_rows], train_df['label'].values[:n_rows])
    val_pred = clf.predict(train_test[n_rows:n_rows+30000])
    print(f1_score(train_df['label'].values[n_rows:n_rows+30000], val_pred, average='macro'))
    print('training time(s):', time()-t0)
    joblib.dump(clf, name+'.pkl')
    
    
def predict(clf):
    test_df = pd.read_csv(test_file_path, sep='\t')
    test_text = tfidf.transform(test_df['text'])
    y = clf.predict(test_text)
    y = pd.DataFrame(columns=['label'], data=y)
    y.to_csv('predict_a.csv', sep='\t', index=False)
    print('prcessing complete')
clf1 = SC(early_stopping=True)
training(clf1, len(train_df)-30000, 'svm')
clf2 = LR(multi_class='multinomial', max_iter=500)
training(clf2, len(train_df)-30000, 'logistic')
predict(clf2)