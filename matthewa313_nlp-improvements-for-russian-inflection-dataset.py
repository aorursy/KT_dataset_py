import os
import pymorphy2
import re
import pandas as pd

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
morph = pymorphy2.MorphAnalyzer()
retoken = re.compile(r'[\'\w\-]+')
def normalize(text):
    text = retoken.findall(text.lower()) # make all text lowercase
    text = [morph.parse(x)[0].normal_form for x in text] # morphological analysis
    return ' '.join(text)
%%time
train['title'] = train['title'].astype(str)
train['description'] = train['description'].astype(str)
test['title'] = test['title'].astype(str)
test['description'] = test['description'].astype(str)
%%time
train['title'] = train['title'].apply(normalize)
train['description'] = train['description'].apply(normalize)
train.to_csv("updnlp-train.csv"); del train
test['title'] = test['title'].apply(normalize)
test['description'] = test['description'].apply(normalize)
print(normalize('собака'))
print(normalize('нет собаки'))
print(normalize('Дай собаке кость.'))
test.to_csv("updnlp-test.csv")