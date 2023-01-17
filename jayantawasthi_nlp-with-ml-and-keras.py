# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load
%matplotlib inline
import re
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from nltk.tokenize import TweetTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import xgboost
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,SpatialDropout1D
from tensorflow.keras.layers import LSTM,Dropout
from keras.layers import Bidirectional
from tensorflow.keras.optimizers import RMSprop,Adam

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train=pd.read_csv("/kaggle/input/twitter-sentiment-analysis-hatred-speech/train.csv")
test=pd.read_csv("/kaggle/input/twitter-sentiment-analysis-hatred-speech/test.csv")
train.head()
test.head()
def drop(te):
    te.drop("id",axis=1,inplace=True)
drop(train)
drop(test)
train.isnull().any()
test.isnull().any()
train["label"].value_counts()
sns.countplot("label",data=train)
tweettoken = TweetTokenizer(strip_handles=True, reduce_len=True)
stemmer=PorterStemmer()
collect=[]
collecttest=[]
def preprocess(t,kpc):
    tee=re.sub('[^a-zA-Z]'," ",t)
    tee=tee.lower()
    res=tweettoken.tokenize(tee)
    for i in res:
        if i in stopwords.words('english'):
            res.remove(i)
    rest=[]
    for k in res:
        rest.append(stemmer.stem(k))
    ret=" ".join(rest)
    if kpc==1:
        collect.append(ret)
    elif kpc==0:
        collecttest.append(ret)
def splitpro(t,q,m):
         for j in range(q):
                 preprocess(t["tweet"].iloc[j],m)
splitpro(train,31962,1)
splitpro(test,17197,0)
len(collect)
len(collecttest)
len(test)
collect[:5]
collecttest[:5]
val=train["label"].values
val
def bow(ll):
    cv=CountVectorizer(max_features=200)
    x=cv.fit_transform(ll).toarray()
    return x
    
y=bow(collect)
y[0]
len(y[0][:])
from imblearn.under_sampling import NearMiss
tt=NearMiss()
x_us,y_us=tt.fit_sample(y,val)
x_us.shape
(x_train,x_test,y_train,y_test) = train_test_split(x_us,y_us, train_size=0.80, random_state=42)
x_train
rnd_clf=RandomForestClassifier(n_estimators=200,random_state=42)
rnd_clf.fit(x_train,y_train)
rnd_clf.score(x_test,y_test)
mm=[300,400,500,600]
for i in mm:
    rnd_clf=RandomForestClassifier(n_estimators=i,random_state=42)
    rnd_clf.fit(x_train,y_train)
    t=rnd_clf.score(x_test,y_test)
    print(t)
    print("*"*40)
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB()
clf.fit(x_train,y_train)
clf.score(x_train,y_train)
params={
    "eta":[0.01,0.2],
    "min_child_weight":[1,2,3,4,5,6,7,8,9,10],
    "max_depth":[3,6,8,10,15,20,25,30],
    "gamma":[0.0,0.1,0.2,0.3,0.4,0.5,0.6],
    "subsample":[0.5,0.6,0.7,0.8,0.9],
    "colsample_bytree":[0.6,0.7,0.8,0.8],
    "reg_alpha":[0,0.001,0.005,0.01,0.05],
    "learning_rate":[0.05,0.10,0.15,0.20,0.25,0.30],
    "max_leaf_nodes":[8,16,24,32,40],
}
from sklearn.model_selection import RandomizedSearchCV
classifier=xgboost.XGBClassifier()
random=RandomizedSearchCV(classifier,param_distributions=params,n_iter=5,scoring="roc_auc",cv=5,verbose=3)
random.fit(x_train,y_train)
random.best_estimator_
classifier=xgboost.XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
              colsample_bynode=1, colsample_bytree=0.6, eta=0.01, gamma=0.1,
              gpu_id=-1, importance_type='gain', interaction_constraints='',
              learning_rate=0.2, max_delta_step=0, max_depth=25,
              max_leaf_nodes=40, min_child_weight=1,
              monotone_constraints='()', n_estimators=100, n_jobs=0,
              num_parallel_tree=1, random_state=0, reg_alpha=0.05, reg_lambda=1,
              scale_pos_weight=1, subsample=0.6, tree_method='exact',
              validate_parameters=1, verbosity=None)
classifier.fit(x_train,y_train)
classifier.score(x_test,y_test)
from sklearn import svm
C = [1,10,20,25,30,35,40,50]
for i in C:
    svc = svm.SVC(kernel='linear', C=i)
    svc.fit(x_train,y_train)
    t=svc.score(x_test,y_test)
    print(t)
for i in C:
    svc = svm.SVC(kernel='rbf', C=i)
    svc.fit(x_train,y_train)
    t=svc.score(x_test,y_test)
    print(t)
oneh=[]
oneht=[]
def hot(cc,k):
    for i in cc:
        if k==1:
            oneh.append(one_hot(i,10000))
        elif k==0:
            oneht.append(one_hot(i,10000))
hot(collect,1)
hot(collecttest,0)
len(oneh[0])
oneh[:1]
len(oneh)
len(oneht)
max=0
for i in oneh:
    tq=len(i)
    if tq>max:
        max=tq
print(max)
sent=40
emoneh=pad_sequences(oneh,padding="pre",maxlen=sent)
emoneht=pad_sequences(oneht,padding="pre",maxlen=sent)
emoneh[:1]
emoneht[:1]
xtrain,xtest,ytrain,ytest=train_test_split(emoneh,val,train_size=0.80,random_state=42)
model=Sequential()
model.add(Embedding(10000,100,input_length=sent))
model.add(SpatialDropout1D(0.4))
model.add(LSTM(100))
model.add(Dropout(0.1))
model.add(Dense(1, activation='softmax'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()
model.fit(xtrain,ytrain,validation_data=(xtest,ytest),epochs=50,batch_size=300,verbose=2)
ytest
val
model.predict(emoneht)