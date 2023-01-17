from importlib import reload

import sys

from imp import reload

import warnings

import pandas as pd
df = pd.read_csv('../input/imdb_master.csv',encoding="latin-1")

df = df.drop(['Unnamed: 0','type','file'],axis=1)

df.columns = ["review","sentiment"]

df.loc[df['sentiment'] == 'neg', 'sentiment'] = 0

df.loc[df['sentiment'] == 'pos', 'sentiment'] = 1

df = df[df.sentiment != 'unsup']

df
import re

from nltk.stem import WordNetLemmatizer

from nltk.corpus import stopwords



stop_words = set(stopwords.words("english")) 

lemmatizer = WordNetLemmatizer()



def clean_text(text):

    text = re.sub(r'[^\w\s]','',text)

    text = text.lower()

    text = [lemmatizer.lemmatize(token) for token in text.split(' ')]

    text = [word for word in text if not word in stop_words]

    text = " ".join(text)

    return text

#for text in df.review:

#    print(clean_text(text))

df['update_review'] = df.review.apply(lambda x: clean_text(x))

def words(df):

    words = set()

    for text in df.update_review:

        words.update(set(text.split(' ')))

    words = dict.fromkeys(words)

    words.pop('')

    return words

W = words(df)

print("pos/neg" in W)
def counter(df, words):

    for i in words:

        words[i] = {}

    words['pos/neg'] = {}

    for i in range(len(df)):

        data = df.iloc[i]

        words['pos/neg'][i] = data.sentiment

        text = data.update_review.split(' ')

        for j in text:

            if j!='':

                if i not in words[j]:

                    words[j][i]=1

                else:

                    words[j][i]+=1

        #if (i+1)%500 == 0:

        #    print(i+1)

    result = {}

    for i in words:

        if len(words[i])>1:

            result[i] = words[i]

        #break

    print(len(result))

    return result

words = counter(df, W)

#print(len(words))

#words.pop('')
import math

def TDF(words):

    k = 0

    for i in words:

        if i!='pos/neg':

            k+=1

            #print(len(words[i]))

            j = words[i].keys()

            a = [0 for k in j if words['pos/neg'][k]==0]

            b = len(words[i])-len(a)

            a = len(a)

            #print(a,b)

            #break

            for j in words[i]:

                if a==0:

                    words[i][j] = words[i][j]

                elif b==0:

                    words[i][j] = -words[i][j]

                else:

                    words[i][j] = words[i][j]*math.log(b/a)

            #print(words[i])

            #if k%500==0:

            #    print(words[i])

                #print(k)

    t = words.copy()

    for i in words:

        t[i] = 0

        for j in words[i]:

            t[i]+=words[i][j]

    return t

a = TDF(words)

a.pop('pos/neg')

words={}

for i in a:

    if a[i]!=0:

        words[i]=a[i]

del a

words
len(words)
from torch.utils.data import TensorDataset,DataLoader

import torch

import torchvision
a = pd.Series(words)

a
val_dataset = TensorDataset(a)

#val_ds, test_ds = torch.utils.data.random_split(val_dataset, [1500, 1500])

#val_loader = DataLoader(val_ds, batch_size=128)

#test_loader = DataLoader(test_ds, batch_size=128)