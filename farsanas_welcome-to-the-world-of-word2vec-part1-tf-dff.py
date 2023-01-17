from IPython.display import YouTubeVideo      

YouTubeVideo('vEmm9fZJuuM')
import pandas as pd

import sklearn as sk

import math #Tocalculate IDF



first= 'Deep Learning is fascinating'

second= 'I am loving Deep Learning'



#split spring into words

first = first.split(" ")

second= second.split(" ")



#to remove duplicate words

total= set(first).union(set(second))

print(total)



wordDictA = dict.fromkeys(total, 0) 

wordDictB = dict.fromkeys(total, 0)

for word in first:

    wordDictA[word]+=1

    

for word in second:

    wordDictB[word]+=1

#Output 

pd.DataFrame([wordDictA, wordDictB])
from sklearn.feature_extraction.text import CountVectorizer

import pandas as pd

sent = ['Deep Learning is is fascinating', 

        'am loving Deep Learning lot'

       ]

vect = CountVectorizer(analyzer= 'word')

sent_vt = vect.fit_transform(sent)



count_tokens = vect.get_feature_names()

df_countvec = pd.DataFrame(data = sent_vt.toarray(),index = ['sentence1', 'sentence2'], columns = count_tokens)

print(df_countvec)
#To calculate Term Freguency

def computeTF(wordDict, bow):

    tfDict = {}

    bowCount = len(bow)

    for word, count in wordDict.items():

        tfDict[word] = count/float(bowCount)

    return tfDict

#running our sentences through the tf function:

tfFirst = computeTF(wordDictA, first)

tfSecond = computeTF(wordDictB, second)

#Converting to dataframe for visualization

tf_df= pd.DataFrame([tfFirst, tfSecond])
tf_df
def computeIDF(docList):

    idfDict = {}

    N = len(docList)

    

    idfDict = dict.fromkeys(docList[0].keys(), 0)

    for doc in docList:

        for word, val in doc.items():

            if val > 0:

                idfDict[word] += 1

    

    for word, val in idfDict.items():

        idfDict[word] = math.log10(N / float(val))

        

    return idfDict

#inputing our sentences in the log file

idfs = computeIDF([wordDictA, wordDictB])

#The actual calculation of TF*IDF from the table above:

def computeTFIDF(tfBow, idfs):

    tfidf = {}

    for word, val in tfBow.items():

        tfidf[word] = val*idfs[word]

    return tfidf

#running our two sentences through the IDF:

idfFirst = computeTFIDF(tfFirst, idfs)

idfSecond = computeTFIDF(tfSecond, idfs)

#putting it in a dataframe

tfidf= pd.DataFrame([idfFirst, idfSecond])
tfidf
from sklearn.feature_extraction.text import TfidfVectorizer



sent = ['Deep Learning is fascinating', 

        'i am loving Deep  Learning'

       ]

    

#vect = TfidfVectorizer(norm = False, smooth_idf = False, analyzer= 'word')

vect = TfidfVectorizer(use_idf = True, smooth_idf = False,vocabulary=None,input='content',norm='l2',

    lowercase=True,preprocessor=None,

    tokenizer=None

                      )

sent_vt = vect.fit_transform(sent)

tfid_tokens = vect.get_feature_names()

df_tfidvec = pd.DataFrame(data = sent_vt.toarray(),index = ['sentence1', 'sentence2'], columns = tfid_tokens)

sent_vect = vect.fit_transform(sent)

print(df_tfidvec)