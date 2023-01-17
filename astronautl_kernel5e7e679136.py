# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from pathlib import Path 

DATA_ROOT = Path("..") / "/kaggle/input/sentiment-analysis-on-movie-reviews"

train = pd.read_csv(DATA_ROOT / 'train.tsv.zip', sep="\t")

test = pd.read_csv(DATA_ROOT / 'test.tsv.zip', sep="\t")

print(train.shape,test.shape)

train.head()
m = train.shape[0]



train_two_gram = []

train_two_gram_label = []



for i in range(m):

    #IDofSentence = train.loc[i,'SentenceId']

    s = train.loc[i, 'Phrase']

    cnt = 0

    for j in s:

        if j == ' ':

            cnt += 1

    if cnt == 1:

        train_two_gram.append(s)

        train_two_gram_label.append(train.loc[i,'Sentiment'])
train_two_gram_dict = {}



for i in train_two_gram:

    if not train_two_gram_dict.get(i):

        train_two_gram_dict[i] = 1

    else:

        train_two_gram_dict[i] += 1



for k,v in train_two_gram_dict.items():

    if v>1:

        print(k)

print('end')



S = len(train_two_gram_dict)

train_x_two = np.identity(S)

tmp = np.array([1]*S)

tmp = np.expand_dims(tmp,axis = 1)

train_x_two = np.concatenate((train_x_two, tmp), axis = 1)

train_y_two = np.zeros([S, 5])



for i in range(S):

    cls = train_two_gram_label[i]

    train_y_two[i][cls] = 1

tmp = np.concatenate((train_x_two, train_y_two), axis=1)

np.random.shuffle(tmp)

tmp_train_two, tmp_valid_two = tmp[S//4:], tmp[:S//4]

train_x_two, train_y_two = np.split(tmp_train_two, [S+1], 1)

valid_x_two, valid_y_two = np.split(tmp_valid_two, [S+1], 1)

np.random.seed(2000)

w2 = np.random.randn(S+1,5,)

#w1 = np.zeros([S+1,5])
m = test.shape[0]

test_sentenceId = []

test_two_gram = []



for i in range(m):

    #IDofSentence = test.loc[i,'SentenceId']

    s = test.loc[i, 'Phrase']

    cnt = 0

    for j in s:

        if j == ' ':

            cnt += 1

    if cnt == 0:

        test_two_gram.append(s)

        test_sentenceId.append(test.loc[i, 'SentenceId'])
test_two_gram_dict = {}



for i in test_two_gram:

    if not test_two_gram_dict.get(i):

        test_two_gram_dict[i] = 1

    else:

        test_two_gram_dict[i] += 1



for k,v in test_two_gram_dict.items():

    if v>1:

        print(k)

print('end')



S_test = len(test_two_gram_dict)

test_x_two = np.identity(S_test)

tmp = np.ones([S_test,S+1-S_test])

test_x_two = np.concatenate( (test_x_two, tmp), axis = 1)
test_x_two[:5]
def cal_y_predict(w, x):

    y_raw = np.dot(x, w)

    s = np.sum(y_raw, axis=1)

    for i in range(y_raw.shape[0]):

        for j in range(y_raw.shape[1]):

            y_raw[i][j] /= s[i]

    return y_raw
x = train_x_two

y = train_y_two

lr = 10    #累次尝试过0.01、0.1、1、10

N = train_x_two.shape[0]

cnt = 1

y_raw = cal_y_predict(w2, train_x_two)
def Lose(y_raw, y):

    loss = 0

    

    cls = np.where(y_raw[0]==np.max(y_raw[0]))



    for i in range(1, y_raw.shape[0]):

        m = np.max(y_raw[i])

        tmp = np.where(y_raw[i]==m)

        cls = np.concatenate( (cls, tmp),axis=1)

    a = cls.shape[0]

    b = cls.shape[1]

    cls.reshape(b,a)

    

    for i in range(y.shape[0]):

        k = y[i][cls[0][i]]

        if k!=1:

            loss += 1

    return loss
loss = Lose(y_raw, train_y_two)
loss_list = [loss]
while cnt<100 and loss > 10:

    cnt += 1

    w2 += lr * (np.dot(x.T, y-y_raw ) ) / N

    y_raw = cal_y_predict(w2, x)

    loss = Lose(y_raw, y)

    print(loss)

    loss_list.append(loss)

    if loss >= loss_list[-2]:

        print(cnt)

        break
x = valid_x_two

y = valid_y_two

lr = 10  #累次尝试过0.01、0.1、1、10

N = valid_x_two.shape[0]

cnt = 1

y_raw = cal_y_predict(w2, x)
loss = Lose(y_raw, y)

loss_list.append(loss)

print(loss)
while cnt<100 and loss > 10:

    cnt += 1

    w2 += lr * (np.dot(x.T, y-y_raw ) ) / N

    y_raw = cal_y_predict(w2, x)

    loss = Lose(y_raw, y)

    print(loss)

    loss_list.append(loss)

    if loss >= loss_list[-2]:

        print(cnt)

        break
PhraseId = np.dot(test_x_two, w2)
tmp = PhraseId

tmp.shape
cls = np.where(tmp[0]==np.max(tmp[0]))

print(cls)

for i in range(1, tmp.shape[0]):

    #print(i)

    m = np.max(tmp[i])

    k = np.where(tmp[i]==m)

    cls = np.concatenate( (cls, k),axis=1)


a = cls.shape[0]

b = cls.shape[1]

cls = np.reshape(cls, (b,a))
cls.shape
cls
len(cls)
a = test['PhraseId']

len(a)
len(test_sentenceId)
cur = test_sentenceId[0]

voter = [0,1,0,0,0]

final = []

final_sentenceId = []

for i in range(1,len(test_sentenceId)):

    if cur == test_sentenceId[i]:

        voter[cls[i][0]]+=1

    else:

        final_sentenceId.append(cur)

        cur = test_sentenceId[i]

        final.append(voter.index(max(voter)))

        voter = [0,0,0,0,0]
len(final_sentenceId)
len(final)
final.pop()
len(final)
b = test['SentenceId']
len(b)
c = np.unique(b)
len(c)
d=np.array(test_sentenceId)
e = np.unique(d)
len(e)
final.append(2)
test_sentenceId[-1]
test.loc[66291, 'SentenceId']
ans = []

for i in range(test.shape[0]):

    if test.loc[i,'SentenceId'] in test_sentenceId:

        if test.loc[i,'SentenceId'] in final_sentenceId:

            k = final_sentenceId.index(test.loc[i,'SentenceId'])  

            ans.append(cls[k][0])

        else:

            ans.append(2)

    else:

        ans.append(2)

ans[:10]
len(ans)
ans = np.array(ans)
a = test['PhraseId']
df = pd.DataFrame({'PhraseId':a,'Sentiment':ans})
df.to_csv("submission.csv",index=False,sep=',')