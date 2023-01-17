import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

sample = pd.read_csv('../input/sample_submission.csv')
import nltk
from nltk.stem import WordNetLemmatizer

wordnet_lemmatizer = WordNetLemmatizer()



from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))



def myTokenizer(s):

    s=s.lower()

    tokens = nltk.tokenize.word_tokenize(s)

    tokens = [t for t in tokens if len(t) > 2]

    tokens = [wordnet_lemmatizer.lemmatize(t) for t in tokens]

    tokens = [t for t in tokens if t not in stop_words]

    return tokens

    
word_index_map={}

curr_index = 0

for index,row in train.iterrows():

    tokens = myTokenizer(row['text'])

    for t in tokens:

        if t not in word_index_map:

            word_index_map[t]=curr_index

            curr_index +=1 

print(len(word_index_map))

N = len(train)

data = np.zeros((N,len(word_index_map)+1)) # +1 for Label

from sklearn import preprocessing
lbl_enc = preprocessing.LabelEncoder()

y = lbl_enc.fit_transform(train.author.values)
def tokens_to_vectors(tokens,label):

    if(len(tokens) == 0):

        print('No Tokens Available')

    x = np.zeros(len(word_index_map)+1) # +1 is for label

    for t in tokens:

        if t in word_index_map:

            x[word_index_map[t]] +=1

    if x.sum() != 0:

        x = x/x.sum()

    x[-1] = label

    return x

for idx, row in train.iterrows():

    tokens = myTokenizer(row['text'])

    if len(tokens) == 0:

        print(row['text'])

    data[idx,:] = tokens_to_vectors(tokens,y[idx])

    
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()

model.fit(data[:,:-1],data[:,-1])
model.score(data[:,:-1],data[:,-1])
N=len(test)

td = np.zeros((N,len(word_index_map)+1))

for idx,row in test.iterrows():

    tokens = myTokenizer(row['text'])

    td[idx,:] = tokens_to_vectors(tokens,0)

td = td[:,:-1]

    

p = model.predict_proba(td)# 0 is dummy

print(p.shape)

    
result = pd.DataFrame()

result['id'] = test['id']

result['EAP'] = p[:,0]

result['HPL'] = p[:,1]

result['MWS'] = p[:,2]

result.head()

result.to_csv("result.csv", index=False)