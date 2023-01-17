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
from kaggle.competitions import twosigmanews
import numpy as np
env = twosigmanews.make_env()
(market_train_df, news_train_df) = env.get_training_data()
# Select small portion of data for the matter of time complexity, 
start='2016-09-09'
M=market_train_df.loc[market_train_df['time']>start]
N=news_train_df.loc[news_train_df['time']>start]
M=M.reset_index(drop=True)
N=N.reset_index(drop=True)

# I need to parse time to get only YYYY-MM-DD
# where the term zman means date
M['zaman']=M.apply(lambda row: str(row["time"]).split()[0],axis=1)
N['zaman']=N.apply(lambda row: str(row["time"]).split()[0],axis=1)

from sklearn.feature_extraction.text import CountVectorizer

size=0
#headlineTag vectorizer Training
headlineT= [s.replace(" ","_").strip() for s in N["headlineTag"]]
headlineTVectorizer = CountVectorizer(token_pattern=r'[a-zA-Z0-9\-\.:_]+')
headlineTVectorizer.fit(headlineT)
q=len(headlineTVectorizer.vocabulary_)
size=size+q
print("headline Tag",q)

# provider vectorizer Training
provider=list(N["provider"])
providerVectorizer=CountVectorizer(token_pattern=r'[a-zA-Z0-9\-\.:_]+')
providerVectorizer.fit(provider)
q=len(providerVectorizer.vocabulary_)
size=size+q
print("Provider",q)


#headline vectorizer
headlines= N["headline"]
headlineVectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=10000, stop_words='english', ngram_range=(1,3), token_pattern=r'[a-zA-Z0-9\-\.:]+')
headlineVectorizer.fit(headlines)
q=len(headlineVectorizer.vocabulary_)
size=size+q
print("Headline",q)

#subjects vectorizer
import re
ss=N['subjects']
subjects=[re.sub ("[{}'',]" ,"",s) for s in ss]
sset=list(set((" ".join(subjects)).split()))
subjectVectorizer = CountVectorizer(token_pattern=r'[a-zA-Z0-9\-\.:_]+')
subjectVectorizer.fit(sset)
q=len(subjectVectorizer.vocabulary_)
size=size+q
print("Subjects",q)

au=N['audiences']
aus=[re.sub ("[{}'',]" ,"",a) for a in au]
saus=list(set((" ".join(aus)).split()))
auVectorizer = CountVectorizer(token_pattern=r'[a-zA-Z0-9\-\.:_]+')
auVectorizer.fit(saus)
q=len(auVectorizer.vocabulary_)
size=size+q
print("Audience",q)

print("* The total for news is", size)
def buildVec(newsQA, marketQA):
    #headline Trasformations
    headlines=" ".join(list(newsQA["headline"]))
    headlinesVec=headlineVectorizer.transform([headlines]).todense()
    # provider
    prov=" ".join(list(newsQA["provider"]))
    provVec=providerVectorizer.transform([prov]).todense()
    #comment
    comment=newsQA["marketCommentary"]
    c=" ".join(list([str(s) for s in comment]))
    commentVec= [[c.split().count("True") , c.split().count("False")]]
    #headlineTag
    headlineT=" ".join([s.replace(" ","_") for s in newsQA["headlineTag"]])
    headlineTVec=headlineTVectorizer.transform([headlineT]).todense()
    #audience transformation
    aud=[re.sub ("[{}'',]" ,"",s) for s in newsQA["audiences"]]
    auVec=auVectorizer.transform([" ".join(aud)]).todense()
    #subject transformation
    subjects=[re.sub ("[{}'',]" ,"",s) for s in newsQA["subjects"]]
    subjectVec= subjectVectorizer.transform( [" ".join(subjects)]).todense()
    #numeric values
    numvar=["urgency", "takeSequence", "bodySize", "companyCount","sentenceCount","wordCount","firstMentionSentence","sentimentClass","sentimentNegative","sentimentNeutral","sentimentPositive","sentimentWordCount","noveltyCount12H","noveltyCount24H","noveltyCount3D","noveltyCount5D","noveltyCount7D","volumeCounts12H","volumeCounts24H","volumeCounts3D","volumeCounts5D","volumeCounts7D"]
    numvarMean=newsQA[numvar].mean()
    numvarMean=np.array(numvarMean)
    # Concatenation NEws variables
    newsVec=np.concatenate((headlinesVec, provVec, commentVec,headlineTVec,auVec, subjectVec, [numvarMean]),1)
    #market vec
    marketVec=np.array(list(marketQA[4:16]))
    # all vec
    allvec= np.concatenate((newsVec,[marketVec]),1)
    return allvec

# train data collection
def buildTrainData(market_train_df, news_train_df, limit):
    X=[]
    i=0
    count=0
    hmnanrow=0
    for row in market_train_df.itertuples():
        if not (True in np.isnan(np.array(row[-10:15]))):
            companyName= row.assetName
            zaman=row.zaman
            newsQA=news_train_df[  (news_train_df['assetName']==companyName) & (news_train_df['zaman']==zaman)]
            newsCount=newsQA.shape[0]
            count=count+1
            if (count % 1000 ==0):
             print(count)
            if (count% (limit+1)==0):
             print(count, len(X), hmnanrow)
             break
            #if newsCount>0:
            if True:
             marketQA=row
             vecc=buildVec(newsQA, marketQA)
             vecc2=np.array(vecc)[0]
             X.append(vecc2)
    return np.array(X)

X_train= buildTrainData(M, N,5000)
X_train[np.isnan(X_train)]=0
# lets traing and measue success as abinary classifications
lastC=X_train.shape[1]-1
lastC
X=X_train[:,:lastC]
y=X_train[:,lastC]

# Standardize the entire matrix
from sklearn.preprocessing import StandardScaler
stdsc = StandardScaler()
X_std = stdsc.fit_transform(X)
X_std.shape

# I convert the problem for binary classification, So not he problem is if the target varible is bigger than ZERO or NOT
ycat= y>0
sum(ycat) /ycat.size
# I apply Log Reg to model, it must be bigger than 0.5147 (which is  class ratio above)
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

clf=LogisticRegression()
scores = cross_val_score(clf,X_std, ycat, cv=5)
print(scores.mean())


