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
import pandas as pd
import numpy as np
import scipy.stats as stats
from sklearn import linear_model, metrics, model_selection, neighbors, ensemble, svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
# HASHING

def StringHash(a, m=257, C=1024):
    
# m represents the estimated cardinality of the items set
# C represents a number that is larger that ord(c)
    hash=0
    for i in range(len(a)):
        hash = (hash * C + ord(a[i])) % m
    return hash 



def multiStringHash(b, m=257, C=1024, lenword=3):
    
# m represents the estimated cardinality of the items set
# C represents a number that is larger that ord(c)
    tweethash =[]
    for a in b:
        hash=0
        for j in range(len(a)):
            hash = (hash * C + ord(a[j])) % m
        tweethash.append(hash) if len(a)>lenword else None
    return tweethash 
#Noves variables 'x': 

import re #regular expressions

def feat_extr(data, mbin=2884, Cbin=1024, mtag=2884, Ctag=1024, lenword=3):

    tags, numtags, hash_tags, hash_bins, party_tags = [], [], [], [], []
    ptags = [["@EnComu_Podem",'@ahorapodemos'],
             ['@CiudadanosCs'],
             ['@Esquerra_ERC'],
             ['@JuntsXCat'],
             ["@PPCatalunya","@PPopular"],
             ["@PSOE","@socialistes_cat"]]

    i=0
    for tweet in data["text"]:

        # '@' and '#':
        ref = re.findall(r"[@#]\w+", tweet) 
        refs = [ref[i][1:] for i in range(len(ref))]
        tags.append(refs)
        sh = multiStringHash(refs, mtag, Ctag, lenword=0)
        hash_tags.append(sh)
        numtags.append(len(ref))

        # '@' and '#' with Party references:
        a = []
        for p in ptags:
            k=0
            for s in p:
                k = 1 if re.search(s, tweet)!=None else k
            a.append(k)
        party_tags.append(a)


        # Not '@' and '#':
        tweetWords = str(tweet).split(sep=" ")
        cumbins = []

        j=0
        for word in tweetWords:

            # HASH # 
            ref = re.search(r"[@#]\w+", word)
            if ref==None:
                if len(word)>lenword:
                    sh = StringHash(word.lower(),  mbin, Cbin)
                    cumbins.append(sh) # Hashing no tags
            j+=1

        hash_bins.append(cumbins)

        i+=1

    # Hash dummies
    refmat = np.zeros((data.shape[0],mtag))
    wordmat = np.zeros((data.shape[0],mbin))
    for i in range(data.shape[0]):

        # '@' or '#'
        for j in range(mtag):
            refmat[i][j] = 1 if j in hash_tags[i] else 0

        # Not '@' or '#'
        for j in range(mbin):
            wordmat[i][j] = 1 if j in hash_bins[i] else 0
    
    party_tags = np.array(party_tags)
    numtags = np.resize(numtags,(data.shape[0],1))
    fav_ret = np.array(data[['favorite_count','retweet_count']])

    return pd.DataFrame(np.concatenate((refmat,party_tags,numtags,fav_ret,wordmat),axis=1))
X_train = feat_extr(train1)
rf = RandomForestClassifier(n_estimators = 750)

x_tr,x_te,y_tr,y_te = model_selection.train_test_split(X_train, train1["party"], test_size = 0.3)

#nn.fit(x_tr, y_tr)
#lr.fit(x_tr, y_tr)
rf.fit(x_tr, y_tr)

yhat = rf.predict(x_te)
metrics.accuracy_score(yhat, y_te)
