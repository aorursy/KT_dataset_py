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
from sklearn.feature_extraction.text import TfidfVectorizer
from  sklearn.naive_bayes import MultinomialNB
frame = pd.read_csv('../input/train.tsv',sep= '\t')
train_data_raw = frame['Phrase']
train_data = [data.replace(',','').replace('.','') for data in train_data_raw]
tfidf = TfidfVectorizer()
train_data = tfidf.fit_transform(train_data)
train_data.shape
test_frame = pd.read_csv('../input/test.tsv',sep= '\t')
test_data_raw = test_frame['Phrase']
test_data = [data.replace(',','').replace('.','') for data in test_data_raw]
test = tfidf.transform(test_data)
model = MultinomialNB(0.01)
label = frame['Sentiment'].values
model.fit(train_data,label)

result = model.predict(test)
ans = []
for id ,emotionId in zip(test_frame['PhraseId'],result):
    tmp = []
    tmp.append(id);tmp.append(emotionId)
    ans.append(tmp)
frame = pd.DataFrame(ans,columns = ['PhraseId','Sentiment'])
frame.to_csv('my.csv',index=False)