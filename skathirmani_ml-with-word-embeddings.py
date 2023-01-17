# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from gensim.models import KeyedVectors
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import nltk
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
data = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
data['question_text'] = data['question_text'].str.lower().str.replace('[^a-z ]', '')
test['question_text'] = test['question_text'].str.lower().str.replace('[^a-z ]', '')
target_zero_rows = data[data['target']==0]
target_one_rows = data[data['target']==1]

target_zero_random_rows = np.random.randint(1, target_zero_rows.shape[0], target_one_rows.shape[0]*2)
df1 = target_zero_rows.iloc[target_zero_random_rows]

train, validate = train_test_split(data, test_size=0.3, random_state=100)
train.shape
# Any results you write to the current directory are saved as output.
df2 = pd.concat([df1, target_one_rows])
train, validate = train_test_split(df2, test_size=0.3, random_state=100)
train.shape, validate.shape

del data
del target_zero_rows
del target_one_rows
model = KeyedVectors.load_word2vec_format('../input/embeddings/GoogleNews-vectors-negative300/GoogleNews-vectors-negative300.bin', binary=True)
#print (model.most_similar('desk'))

uniq_words = set()
train['question_text'].str.split(' ').apply(uniq_words.update)
uniq_words = list(uniq_words)
len(uniq_words)

stopwords = nltk.corpus.stopwords.words('english')
words = set(uniq_words) - set(stopwords)
words = list(words)
len(words)

list(uniq_words)[:5]
wordvectors = {}
for word in uniq_words:
    try:
        wordvectors[word] = model[word]
    except:
        pass
len(wordvectors)

del model
import gc
gc.collect()
def sent_vectorizer(sent, model):
    sent_vec =[]
    numw = 0
    for w in sent:
        try:
            if numw == 0:
                sent_vec = model[w]
            else:
                sent_vec = np.add(sent_vec, model[w])
            numw+=1
        except:
            pass
    if len(sent_vec)==300:
        return list(np.asarray(sent_vec) / numw)
    else:
        return np.zeros(300)
train_embeddings = []
for question in train['question_text']:
    train_embeddings.append(sent_vectorizer(question.split(' '), wordvectors))
    
test_embeddings = []
for question in test['question_text']:
    test_embeddings.append(sent_vectorizer(question.split(' '), wordvectors))

validate_embeddings = []
for question in validate['question_text']:
    validate_embeddings.append(sent_vectorizer(question.split(' '), wordvectors))
    
min_l = 0
for i in train_embeddings:
    if len(i)==0:
        min_l = min_l + 1
min_l
model = AdaBoostClassifier(n_estimators=800)
model.fit(np.array(train_embeddings), train['target'])

pred = model.predict(validate_embeddings)
from sklearn.metrics import accuracy_score, f1_score
print(accuracy_score(validate['target'], pred))
print(f1_score(validate['target'], pred))
pred_test = model.predict(test_embeddings)
predictions = pd.DataFrame({'qid': test['qid'],
                            'prediction': pred_test})
predictions.head()
predictions['prediction'].value_counts()
predictions.to_csv('submission.csv', index=False)