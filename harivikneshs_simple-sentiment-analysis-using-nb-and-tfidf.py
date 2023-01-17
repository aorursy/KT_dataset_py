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
df=pd.read_csv('../input/training.1600000.processed.noemoticon.csv',encoding="ISO-8859-1", names=["target", "ids", "date", "flag", "user", "text"])

df.drop(['ids','flag','date','user'],axis=1,inplace=True)

df['target']=df['target'].apply(lambda x: x/4)





#following steps are useless i have earlier used them to train using smaller subsets of data by making sure it contains both targets

d1=df[df['target']==1.0]

d2=df[df['target']==0.0]

df=pd.concat([d1,d2])

print(df.shape,df['target'].unique().shape,df[df['target']==1].shape)
from nltk.stem import PorterStemmer

from nltk.corpus import stopwords

stemmer=PorterStemmer()

def filtr(st):

    st=st.lower()

    rs=[x for x in st.split() if x not in stopwords.words("english")]

    rs=[x for x in rs if (x[0]!='@' and x[:5]!='http')]

    rs=[stemmer.stem(word=x) for x in rs]

    return ' '.join(rs)



    
#df['text'].apply(filtr) --running this may improve performance but takes longer time
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.model_selection import train_test_split

xtrain,xtest,ytrain,ytest=train_test_split(df['text'],df['target'],test_size=0.1,shuffle=True)
tv=TfidfVectorizer(ngram_range=(1,2),max_features=10000,stop_words='english') #limit feature size as you may get lot of features

tv.fit(xtrain)
from sklearn.naive_bayes import MultinomialNB

from sklearn.svm import SVC

nb=MultinomialNB(1.5,fit_prior=False)

xtrain1=tv.transform(xtrain)

nb.fit(xtrain1,ytrain)

pred1=nb.predict(tv.transform(xtest))

from sklearn.metrics import confusion_matrix,accuracy_score

print(accuracy_score(ytest,pred1))

print(confusion_matrix(ytest,pred1))
from sklearn.model_selection import GridSearchCV

gcv=GridSearchCV(nb,{'alpha':[1.5,2,3,4,10,100,1.0,0.1,0.001,0.0001],'fit_prior':[True,False]})

gcv.fit(xtrain1,ytrain)
print(gcv.best_score_,gcv.best_params_)

#update hypermeters if you find them great
def result(x):

    x=filtr(x)

    return nb.predict(tv.transform([x]))
result('nlp is super cool') #example sentence, 0-negative 1-positive