# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

ds = pd.read_csv('../input/train.csv',header=0)

ds.info()
ds.dropna(inplace=True)

ds.info()
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer



l1 = ds[['qid1','question1']].copy()

l2 = ds[['qid2','question2']].copy()

lid = ds[['qid2','qid1']].copy()

l = np.vstack([l1,l2])

test_pare= ds.values[0,3:5]

print (test_pare)
vectorizer = CountVectorizer(stop_words='english')

mx = vectorizer.fit(test_pare)

mx
ds_stop_words = ds.copy()

test= ds_stop_words.values[0:5,3:5]

print (test)

def clear(word):

    vectorizer = CountVectorizer(stop_words='english')

    mx = vectorizer.fit([word])

    vectorizer.get_feature_names().sort()

    return vectorizer.get_feature_names()

for t in range(test.shape[0]):

     test[t][0] = clear(test[t][0])

     test[t][1] = clear(test[t][1])

print (test)    
import Levenshtein

words = ['hybridization ']

words2 = ['Hybrid','hybridziationesss']

for t in words:

    for t2 in words2:

        q = Levenshtein.jaro_winkler(t,t2)

        print (t, t2,  q)
def similarity(sentance1, sentance2):

    len1 = len(sentance1)

    len2 = len(sentance2)

    

    difs1=[]

    difs2=[]

    ident = 0

    for s1 in sentance1:

        founded = False

        for s2 in sentance2:

           q = Levenshtein.jaro_winkler(s1,s2)

           if q>.95:

                founded = True

                sentance2.remove(s2)

                break

        

        if founded:

            ident +=1

        else:

           difs1.append(s1)

    differ = len(difs1)+len(sentance2)

    print (ident, differ, np.median([len1,len2]))

    median = np.median([len1,len2])

    differ = median - ident

    res = (np.power(ident,2)-np.power(differ,2))/np.power(median,2)

    

    return np.maximum(res,0)
print (test)

for t in test:

    #print (t[0],t[1])

    print (similarity(t[0],t[1]))

    
ds['predict'][:0,] = 1

ds.head()

#for t in range(ds.values.shape[0]):

#    ds['predict'][t] = similarity(t[3],t[4])

#    if t%10000== 0:

#        print (t)

    