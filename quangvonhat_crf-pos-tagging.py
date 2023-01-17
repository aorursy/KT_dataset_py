import nltk, re, pprint
import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt
import seaborn as sns
import pprint, time
import random
from sklearn.model_selection import train_test_split
from nltk.tokenize import word_tokenize
from sklearn_crfsuite import CRF
from sklearn_crfsuite import metrics
from sklearn_crfsuite import scorers
from collections import Counter

!pip install sklearn_crfsuite

data = pd.read_csv("../input/abcdef/abc.csv")
data.head(50)
sentences=[]
i=0
while i<len(data):
    tmp=[]
    while (data.iloc[i,0])!="." and data.iloc[i,1] !='.':
        tmp.append(data.iloc[i,0])
        i=i+1
    tmp.append('.')
    sentences.append(np.array(tmp))
    i=i+1
tagged_sentences=[]
i=0
while i<len(data):
    tmp=[]
    while (data.iloc[i,0])!="." and data.iloc[i,1] !='.':
        tmp.append(data.iloc[i,1])
        i=i+1
    tmp.append('.')
    tagged_sentences.append(np.array(tmp))
    i=i+1
print(len(tagged_sentences)),
    
print(len(sentences))
train_sentences, test_sentences, train_tags, test_tags = train_test_split(sentences, tagged_sentences, test_size=0.2)
def features(sentence,index):
    return {
        'is_first_capital':int(sentence[index][0].isupper()),
        'is_first_word': int(index==0),
        'is_last_word':int(index==len(sentence)-1),
        'is_complete_capital': int(sentence[index].upper()==sentence[index]),
        'prev_word':'' if index==0 else sentence[index-1],
        'next_word':'' if index==len(sentence)-1 else sentence[index+1],
        'is_numeric':int(sentence[index].isdigit()),
        'is_alphanumeric': int(bool((re.match('^(?=.*[0-9]$)(?=.*[a-zA-Z])',sentence[index])))),
        'prefix_1':sentence[index][0],
        'prefix_2': sentence[index][:2],
        'prefix_3':sentence[index][:3],
        'prefix_4':sentence[index][:4],
        'suffix_1':sentence[index][-1],
        'suffix_2':sentence[index][-2:],
        'suffix_3':sentence[index][-3:],
        'suffix_4':sentence[index][-4:],
        'word_has_hyphen': 1 if '-' in sentence[index] else 0
        
        
    }
def prepareData(tagged_sentences):
    X=[]
    for sentences in tagged_sentences:
        X.append([features(sentences, index) for index in range(len(sentences))])
    return X
X_train = prepareData(train_sentences)
y_train = train_tags
X_test = prepareData(test_sentences)
y_test =  test_tags

crf = CRF(
    algorithm='lbfgs',
    c1=0.01,
    c2=0.1,
    max_iterations=100,
    all_possible_transitions=True
)
crf.fit(X_train, y_train)
X_train[1]
y_train[1]
y_pred=crf.predict(X_test)
y_pred_train=crf.predict(X_train)
#F1 score test
metrics.flat_f1_score(y_test, y_pred,average='weighted',labels=crf.classes_)
#F1 score train
metrics.flat_f1_score(y_train, y_pred_train,average='weighted',labels=crf.classes_)
#Accuracy score test
metrics.flat_accuracy_score(y_test,y_pred)
#Accuracy score train
metrics.flat_accuracy_score(y_train,y_pred_train)
print(metrics.flat_classification_report(
    y_test, y_pred, labels=crf.classes_, digits=3
))


