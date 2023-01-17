# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train= pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')

test= pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')

submission= pd.read_csv('/kaggle/input/nlp-getting-started/sample_submission.csv')
import re

#get the hashtags

train['hashtags']=[re.findall(r"#(\w+)", i) for i in train.text]
#arrange and score the hashtags as per woe

#all the hashtags with occurance and target conversions

from tqdm import tqdm



hts={}

ht=pd.DataFrame()

for i in tqdm(train.hashtags.values):

    #print(i)

    if len(i)>0:

        for j in i:

            #print(j)

            if j not in hts:

                hts[j]=1

                

            else:

                hts[j]+=1

                

ht=pd.DataFrame({"hashtags":[i for i in hts.keys() ], "count":[i for i in hts.values() ]})

ht['targets']=[[j for i,j in zip(train.hashtags, train.target) if k in i] for k in ht.hashtags]

ht['targets']=[sum(i) for i in ht.targets]

ht['pc']=ht['targets']/ht['count']
train['hashtag_score']=0

for i in tqdm(range(0,len(train))):

    #print(i)

    for j in train['hashtags'].iloc[i]:

        train['hashtag_score'].iloc[i]+=ht['pc'][ht.hashtags==j].values[0]

        

train['hashtag_score_normalized']=[i/len(j) if len(j)>0 else 0 for i,j in zip(train['hashtag_score'],train['hashtags'])]

#same for test 

test['hashtags']=[re.findall(r"#(\w+)", i) for i in test.text]

test['hashtag_score']=0

for i in tqdm(range(0,len(test))):

    for j in test['hashtags'].iloc[i]:

        try:

            val=ht['pc'][ht.hashtags==j].values[0]

        except:

            val=0

        test['hashtag_score'].iloc[i]+=val

        

test['hashtag_score_normalized']=[i/len(j) if len(j)>0 else 0 for i,j in zip(test['hashtag_score'],test['hashtags'])]
#get score from keywords

#keyword

keyword={}

for i in train.groupby('keyword'):

    keyword[i[0]]=(round(i[1].target.sum()/len(i[1]),2))







train['keyword_score']=[keyword[i] if i in keyword else 0 for i in train['keyword']]



keyword= train[['keyword','keyword_score']]

keyword=keyword.drop_duplicates()
#score for test



test['keyword_score']=[keyword[keyword.keyword==i]['keyword_score'].values[0] if len(keyword[keyword.keyword==i]['keyword_score']>0) else 0 for i in test['keyword']]

#location

#location



train['location'][train['location'].isnull()]=""

train['location']=train['location'].str.replace(r'[^A-Za-z]+',' ')

train['location']=train['location'].str.lower()

train['location']=[i.strip() for i in train['location']]

loc= set([i for i in train['location']])

loc=[i.strip() for i in loc]



#test

test['location'][test['location'].isnull()]=""

test['location']=test['location'].str.replace(r'[^A-Za-z]+',' ')

test['location']=test['location'].str.lower()

test['location']=[i.strip() if i in loc else '' for i in test['location']]
#clean text

train['text_new']=train['text'].str.lower()

train['text_new']=train['text_new'].str.replace(r'[^a-z]+'," ")

train['text_new']=[i.strip() for i in train['text_new']]

#train['text_new']



test['text_new']=test['text'].str.lower()

test['text_new']=test['text_new'].str.replace(r'[^a-z]+'," ")

test['text_new']=[i.strip() for i in test['text_new']]

#sentence embeddings

!pip install tensorflow-text==2.0.0 --user

import tensorflow as tf

import tensorflow_hub as hub

import tensorflow_text as textb
use = hub.load("https://tfhub.dev/google/universal-sentence-encoder-multilingual-large/3")
X_train = []

for r in tqdm(train.text_new.values):

  emb = use(r)

  review_emb = tf.reshape(emb, [-1]).numpy()

  X_train.append(review_emb)



X_train = np.array(X_train)

y_train = train.target.values



X_test = []

for r in tqdm(test.text_new.values):

  emb = use(r)

  review_emb = tf.reshape(emb, [-1]).numpy()

  X_test.append(review_emb)



X_test = np.array(X_test)
from sklearn.model_selection import train_test_split

train_arrays, test_arrays, train_labels, test_labels = train_test_split(X_train,

                                                                        y_train,

                                                                        random_state =42,

                                                                        test_size=0.20)
from sklearn.svm import SVC

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import train_test_split



from tqdm import tqdm

def svc_param_selection(X, y, nfolds):

    Cs = [1.07]

    gammas = [2.075]

    param_grid = {'C': Cs, 'gamma' : gammas}

    grid_search = GridSearchCV(SVC(kernel='rbf'), param_grid, cv=nfolds, n_jobs=8)

    grid_search.fit(X, y)

    grid_search.best_params_

    return grid_search



model = svc_param_selection(train_arrays,train_labels, 5)
pred = model.predict(test_arrays)

from sklearn.metrics import accuracy_score



accuracy = accuracy_score(test_labels,pred)

accuracy

#on full model

model_final = svc_param_selection(X_train,y_train, 5)

pred=model_final.predict(X_test)



from collections import Counter

Counter(pred)
#improve on by usingg hashtag score and leyword score

test['pred']=pred.tolist()

test['pred']=[1 if i>=.75 else j for i,j in zip(test['hashtag_score_normalized'],test['pred'])]

test['pred']=[1 if i>=.90 else j for i,j in zip(test['keyword_score'],test['pred'])]

test.pred.value_counts()

output = pd.DataFrame({'id': test.id, 'target': test.pred})

output.to_csv('my_submission_hub.csv', index=False)

print("Your submission was successfully saved!")
