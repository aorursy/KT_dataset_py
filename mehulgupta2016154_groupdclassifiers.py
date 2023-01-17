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
x.columns
x=x.drop(['ingredients','separated_ing','cleared_ing','separated_ing_stemmed'],axis=1)
y=y.drop(['ingredients','separated_ing','cleared_ing','separated_ing_stemmed'],axis=1)
lists=list(ENGLISH_STOP_WORDS)+stopwords.words()
from sklearn.feature_extraction.text import TfidfVectorizer  as tfidf,CountVectorizer as cv
z=x['cuisine']
tfidf1=tfidf(max_df=0.9,stop_words=lists,analyzer=u'word')
train=tfidf1.fit_transform(x['separated_ing_lemma'])
test=tfidf1.transform(y['separated_ing_lemma'])
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV as gsc
from sklearn.ensemble import RandomForestClassifier
from xgboost.sklearn import XGBClassifier as xgb
from lightgbm import LGBMClassifier as lgb
from sklearn.linear_model import LogisticRegression as lr
svm={'C':[6]}

from sklearn.preprocessing import LabelEncoder as le
p=le().fit(z)
z=p.transform(z)
from sklearn.model_selection import train_test_split as tts
xtrain,xtest,ztrain,ztest=tts(train,z,train_size=0.7)
r1=lgb(n_estimators=500,max_depth=7,objective='multiclass',metric='multi_logloss',num_classes=20,bagging_fraction=0.6,feature_fraction=0.6)
from sklearn.model_selection import GridSearchCV as gsc
a=gsc(lr(),svm)
from sklearn.neighbors import KNeighborsClassifier as knn
k={'n_neighbors':[5,7,9]}
k1=gsc(knn(),k)
from sklearn.multiclass import OneVsRestClassifier as orc
from sklearn.ensemble import VotingClassifier as vc

v=vc(estimators=[('lr',a),('k1',k1),('lg',r1)],voting='soft')
v.fit(xtrain,ztrain)
from sklearn.metrics import accuracy_score 
print(accuracy_score(ztest,v.predict(xtest)))
z1=v.predict(test)
z=p.inverse_transform(z1)
ff=pd.DataFrame(z,index=y['id'],columns=['cuisine'])
ff.index.name='id'
ff.to_csv('aagya.csv')
