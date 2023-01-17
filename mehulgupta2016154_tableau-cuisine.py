# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
x=pd.read_json('../input/train.json')
y=pd.read_json('../input/test.json')
from nltk.corpus import stopwords
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
import string
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer
z=x['cuisine']
from sklearn.preprocessing import LabelEncoder as le
from sklearn.preprocessing import OneHotEncoder as ohe

x.info()
x.head()
y.info()
y.head()
x['separated_ing']=x['ingredients'].map(lambda x: ' '.join(x))
y['separated_ing']=y['ingredients'].map(lambda x: ' '.join(x))

import string,re
def purify(f):
    f=f.lower()
    f=re.sub('[%s]' % re.escape(string.punctuation),'',f)
    f=re.sub('\s+',' ',f)
    return f
x['cleared_ing']=x['separated_ing'].map(lambda g :purify(g))
y['cleared_ing']=y['separated_ing'].map(lambda g :purify(g))
sb=SnowballStemmer('english')
def stemmer(f):
    lists=[sb.stem(c) for c in f.split(" ")]
    return lists
l=WordNetLemmatizer()
def lemmar(f):
    lists=[l.lemmatize(g) for g in f.split(" ")]
    return lists
x['separated_ing_stemmed']=[stemmer(l) for l in x['cleared_ing']]
x['separated_ing_stemmed']=x['separated_ing_stemmed'].map(lambda x: ' '.join(x))
x['separated_ing_lemma']=[lemmar(l) for l in x['separated_ing_stemmed']]
x['separated_ing_lemma']=x['separated_ing_lemma'].map(lambda x: ' '.join(x))
y['separated_ing_stemmed']=[stemmer(l) for l in y['cleared_ing']]
y['separated_ing_stemmed']=y['separated_ing_stemmed'].map(lambda x: ' '.join(x))
y['separated_ing_lemma']=[lemmar(l) for l in y['separated_ing_stemmed']]
y['separated_ing_lemma']=y['separated_ing_lemma'].map(lambda x: ' '.join(x))
x=x.drop(['ingredients','separated_ing','cleared_ing','separated_ing_stemmed'],axis=1)
y=y.drop(['ingredients','separated_ing','cleared_ing','separated_ing_stemmed'],axis=1)
x.columns
a=''
def f(x):
    global a
    a+=x
for c,d in x.iterrows():
    f(d['separated_ing_lemma'])
a=[x for x in a.split(" ")]
from sklearn.preprocessing import LabelEncoder as le
c=le().fit_transform(a)
c
p=[]
for x1 in x['cuisine']:
    if x1 not in p:
        p.append(x1)
dic={}
for x1 in p:
    dic[x1]=[]
def gg(x):
    dic[x['cuisine']]=list(dic[x['cuisine']])+[f for f in x['separated_ing_lemma'].split(" ") if f not in dic[x['cuisine']]]
for e,d in x.iterrows():   
    gg(d)
    
    
xx=pd.DataFrame(index=p,columns=p)
b=[]
for n in a:
    if n not in b:
        b.append(n)
        
xx1=xx
for z in p:
    for z1 in p:
        xx.loc[z,z1]=len(list(set(dic[z]) & set(dic[z1])))/len(list(set(dic[z]) | set(dic[z1])))
        xx.loc[z1,z]=xx.loc[z,z1]
xx.to_csv("cor.csv")
