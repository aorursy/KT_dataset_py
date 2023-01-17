import numpy as np
import pandas as pd
from sklearn import *
from collections import Counter

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
sub = pd.read_csv('../input/sample_submission.csv').rename(columns={'Predicted': 'Predicted2'})
otrain = datasets.fetch_20newsgroups(subset='train')
otrain = pd.DataFrame({'Id': otrain['filenames'], 'text': otrain['data'], 'target': otrain['target']})
otrain['Id'] = otrain['Id'].map(lambda x: x.split('/')[-1])
otest = datasets.fetch_20newsgroups(subset='test')
otest = pd.DataFrame({'Id': otest['filenames'], 'text': otest['data'], 'target': otest['target']})
otest['Id'] = otest['Id'].map(lambda x: x.split('/')[-1])
train.shape, otrain.shape, test.shape, otest.shape
otrain_ = []
for i in range(len(otrain)):
    t = str(otrain['text'][i]).replace('\n', '\n ')
    for j in range(0, len(t), 300):
        otrain_.append([otrain['Id'][i] + '_' + str(int(j/300)).zfill(3), t[j: j+300], otrain['target'][i]])
otrain = pd.DataFrame(otrain_, columns=['Id', 'text', 'target'])
otest_ = []
for i in range(len(otest)):
    t = str(otest['text'][i]).replace('\n', '\n ')
    for j in range(0, len(t), 300):
        otest_.append([otest['Id'][i] + '_' + str(int(j/300)).zfill(3), t[j: j+300], otest['target'][i]])
otest = pd.DataFrame(otest_, columns=['Id', 'text', 'target'])
otrain = pd.concat((otrain, otest)).reset_index(drop=True)
len(otrain), len(train) + len(test)
train_d = {}
pattern = {}
train_d[0] = Counter(' '.join(otrain['text'].astype(str).values))
pattern[0] = ''.join([c for c, v in train_d[0].most_common(100)])
print(0, repr(pattern[0]))

for i in range(1,5):
    train_d[i] = train[train['difficulty']==i].copy()
    train_d[i] = Counter(' '.join(train_d[i]['ciphertext'].astype(str).values))
    pattern[i] = ''.join([c for c, v in train_d[i].most_common(100)])
    print(i, repr(pattern[i]))
def SubEnc(s, level):
    s = str(s)
    for i in range(level,0, -1):
        s1 = pattern[i]
        s2 = pattern[i-1]
        SubEnc_ = str.maketrans(s1, s2)
        s = s.translate(SubEnc_)
    return s
for i in range(1,5):
    train['ciphertext'] = train.apply(lambda r: SubEnc(r['ciphertext'], i) if r['difficulty'] == i else r['ciphertext'], axis=1)
    test['ciphertext'] = test.apply(lambda r: SubEnc(r['ciphertext'], i) if r['difficulty'] == i else r['ciphertext'], axis=1)
train_d1 = train[train['difficulty']==1].copy()
train_d1['len'] = train_d1['ciphertext'].map(len)
train_d1[train_d1['len']==224].head()
otrain['len'] = otrain['text'].map(len)
otrain[((otrain['text'].str.contains('not provided a service for the co')) & (otrain['target']==18))].head()
#Use a word dictionary to fine tune the mappings
spelling_dict = Counter(' '.join(otrain['text'].astype(str).values).split(' '))
#[w for w in train_d1[train_d1['len']==224]['ciphertext'].values[0].split(' ') if w not in spelling_dict]
#set([w for w in otrain[((otrain['text'].str.contains('not provided a service for the co')) & (otrain['target']==18))]['text'].values[0].split(' ')])
results = []
for d in range(1,5):
    train_d = train[train['difficulty']==d].reset_index(drop=True)
    test_d = test[test['difficulty']==d].reset_index(drop=True)
    tfidf = feature_extraction.text.TfidfVectorizer(analyzer = 'char_wb', ngram_range=(1, 7), lowercase=False) 
    tfidf.fit(pd.concat((train_d['ciphertext'], test_d['ciphertext'])))
    trainf = tfidf.transform(train_d['ciphertext'])
    testf = tfidf.transform(test_d['ciphertext'])
    clf = linear_model.LogisticRegression(tol=0.001, C=10.0, random_state=0, solver='sag', max_iter=90, multi_class='auto', n_jobs=-1)
    clf.fit(trainf, train_d['target'])
    print(d, metrics.accuracy_score(train_d['target'], clf.predict(trainf)))
    test_d['Predicted'] = clf.predict(testf)
    results.append(test_d)
test = pd.concat(results)
sub = pd.merge(sub,test, how='left', on=['Id'])
sub[['Id','Predicted']].to_csv('submission.csv', index=False)
for i in range(1,5):
    print(i, repr(SubEnc('V8g{9827\$A${?^*?}$$v7*.yig$w9.8}', i)))